"""Юнит-тесты longform-логики GigaAMEngine без реальной модели.

Движок собираем через object.__new__ и инъекцию фейков: фейковая gigaam-модель
(forward/_decode/_dtype), мок decode_to_int16_16k_mono и speech_intervals.
merge_intervals_to_chunks работает по-настоящему (она чистая и протестирована
отдельно). Проверяем: сборку ASRResult, склейку текста, сдвиг/округление
таймстемпов слов, батчинг, обработку «нет речи» и роутинг по длительности.
"""

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

import gigaam_api.asr.gigaam_engine as gigaam_engine
from gigaam_api.asr.engine import (
    ASRResult,
    AudioTooLongError,
    InferenceCancelledError,
    SegmentTS,
    WordTS,
)
from gigaam_api.asr.gigaam_engine import GigaAMEngine, _collate
from gigaam_api.config import Settings


class _FakeASRModel:
    """Фейк gigaam-модели: forward возвращает фиктивный encoded, _decode отдаёт
    заранее заданные (text, words) по числу элементов батча."""

    def __init__(self, decode_results: list[tuple[str, list[object] | None]]) -> None:
        self._decode_results = list(decode_results)
        self._dtype = torch.float32
        self.forward_calls = 0
        self.batch_sizes: list[int] = []

    def forward(
        self, wav_pad: torch.Tensor, wav_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        self.batch_sizes.append(int(wav_pad.shape[0]))
        return torch.zeros((wav_pad.shape[0], 1, 1)), wav_lens

    def _decode(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        wav_lens: torch.Tensor,
        word_timestamps: bool,
    ) -> list[tuple[str, list[object] | None]]:
        n = int(wav_lens.shape[0])
        return [self._decode_results.pop(0) for _ in range(n)]


def _bare_engine(settings: Settings, model: object) -> GigaAMEngine:
    eng = object.__new__(GigaAMEngine)
    eng.model_name = settings.MODEL
    eng.device = "cpu"
    eng._model = model
    eng._vad = object()
    eng._settings = settings
    return eng


def _word(text: str, start: float, end: float) -> object:
    return SimpleNamespace(text=text, start=start, end=end)


def _recorder(calls: list[str], label: str, ret: ASRResult) -> Callable[..., ASRResult]:
    """Стаб-метода, записывающий факт вызова и возвращающий фикс. ASRResult."""

    def _stub(*args: object, **kwargs: object) -> ASRResult:
        calls.append(label)
        return ret

    return _stub


# ---------------------------------------------------------------- _collate


def test_collate_pads_to_max_and_reports_lengths() -> None:
    a = torch.ones(3)
    b = torch.ones(5)
    batch, lengths = _collate([a, b])
    assert batch.shape == (2, 5)
    assert lengths.tolist() == [3, 5]
    assert batch[0, 3:].abs().sum().item() == 0.0  # хвост короткого западден нулями
    assert batch[1].abs().sum().item() == 5.0


# ----------------------------------------------------------- longform-сборка


def test_longform_assembles_segments_and_joins_text(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(MODEL="v3_ctc", BATCH_SIZE=4)
    model = _FakeASRModel([("раз", None), ("два", None), ("три", None)])
    eng = _bare_engine(settings, model)

    # 60с тишины (int16); реальный VAD не зовём — speech_intervals замокан.
    monkeypatch.setattr(
        gigaam_engine,
        "decode_to_int16_16k_mono",
        lambda path: torch.zeros(60 * 16000, dtype=torch.int16),
    )
    monkeypatch.setattr(
        gigaam_engine,
        "speech_intervals",
        lambda wav, vad, *, threshold: [(0.0, 18.0), (19.0, 37.0), (38.0, 56.0)],
    )

    result = eng._transcribe_longform("x.wav", word_timestamps=False)

    assert isinstance(result, ASRResult)
    assert result.duration == pytest.approx(60.0)
    assert result.text == "раз два три"
    assert [(s.start, s.end) for s in result.segments] == [(0.0, 18.0), (19.0, 37.0), (38.0, 56.0)]
    assert all(s.words is None for s in result.segments)


def test_longform_shifts_and_rounds_word_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(MODEL="v3_ctc", BATCH_SIZE=4)
    # Слова с ЛОКАЛЬНЫМИ таймстемпами (от начала чанка) — движок сдвигает на seg_start.
    model = _FakeASRModel(
        [
            ("раз", [_word("раз", 0.1234, 0.5)]),
            ("два", [_word("два", 0.2, 0.6)]),
        ]
    )
    eng = _bare_engine(settings, model)
    monkeypatch.setattr(
        gigaam_engine,
        "decode_to_int16_16k_mono",
        lambda path: torch.zeros(40 * 16000, dtype=torch.int16),
    )
    monkeypatch.setattr(
        gigaam_engine,
        "speech_intervals",
        lambda wav, vad, *, threshold: [(0.0, 18.0), (19.0, 37.0)],
    )

    result = eng._transcribe_longform("x.wav", word_timestamps=True)

    seg0, seg1 = result.segments
    assert seg0.words == [WordTS(text="раз", start=0.123, end=0.5)]  # сдвиг 0 + округление до 3
    assert seg1.words == [WordTS(text="два", start=19.2, end=19.6)]  # сдвиг +19.0


def test_longform_batches_by_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(MODEL="v3_ctc", BATCH_SIZE=2)
    model = _FakeASRModel([("a", None), ("b", None), ("c", None)])
    eng = _bare_engine(settings, model)
    monkeypatch.setattr(
        gigaam_engine,
        "decode_to_int16_16k_mono",
        lambda path: torch.zeros(60 * 16000, dtype=torch.int16),
    )
    monkeypatch.setattr(
        gigaam_engine,
        "speech_intervals",
        lambda wav, vad, *, threshold: [(0.0, 18.0), (19.0, 37.0), (38.0, 56.0)],
    )

    eng._transcribe_longform("x.wav", word_timestamps=False)

    # 3 чанка при BATCH_SIZE=2 → батчи [2, 1].
    assert model.forward_calls == 2
    assert model.batch_sizes == [2, 1]


def test_longform_no_speech_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(MODEL="v3_ctc")
    model = _FakeASRModel([])
    eng = _bare_engine(settings, model)
    monkeypatch.setattr(
        gigaam_engine,
        "decode_to_int16_16k_mono",
        lambda path: torch.zeros(60 * 16000, dtype=torch.int16),
    )
    monkeypatch.setattr(gigaam_engine, "speech_intervals", lambda wav, vad, *, threshold: [])

    result = eng._transcribe_longform("x.wav", word_timestamps=False)

    assert result.text == ""
    assert result.segments == []
    assert result.duration == pytest.approx(60.0)
    assert model.forward_calls == 0  # инференс не запускался


# ---------------------------------------------------------------- роутинг


def test_transcribe_routes_short_for_short_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _bare_engine(Settings(MODEL="v3_ctc"), _FakeASRModel([]))
    monkeypatch.setattr(gigaam_engine, "probe_duration", lambda path: 10.0)
    calls: list[str] = []
    monkeypatch.setattr(
        eng, "_transcribe_short", _recorder(calls, "short", ASRResult("", 10.0, []))
    )
    monkeypatch.setattr(
        eng, "_transcribe_longform", _recorder(calls, "long", ASRResult("", 10.0, []))
    )

    eng.transcribe("x.wav", word_timestamps=False)
    assert calls == ["short"]


def test_transcribe_routes_longform_for_long_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _bare_engine(Settings(MODEL="v3_ctc"), _FakeASRModel([]))
    monkeypatch.setattr(gigaam_engine, "probe_duration", lambda path: 30.0)
    calls: list[str] = []
    monkeypatch.setattr(
        eng, "_transcribe_short", _recorder(calls, "short", ASRResult("", 30.0, []))
    )
    monkeypatch.setattr(
        eng, "_transcribe_longform", _recorder(calls, "long", ASRResult("", 30.0, []))
    )

    eng.transcribe("x.wav", word_timestamps=False)
    assert calls == ["long"]


def test_transcribe_raises_when_exceeding_max_audio_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _bare_engine(Settings(MODEL="v3_ctc", MAX_AUDIO_SECONDS=20), _FakeASRModel([]))
    monkeypatch.setattr(gigaam_engine, "probe_duration", lambda path: 30.0)

    with pytest.raises(AudioTooLongError):
        eng.transcribe("x.wav", word_timestamps=False)


def test_longform_cancels_between_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(MODEL="v3_ctc", BATCH_SIZE=1)
    model = _FakeASRModel([("раз", None)])  # хватит на один обработанный батч
    eng = _bare_engine(settings, model)
    monkeypatch.setattr(
        gigaam_engine,
        "decode_to_int16_16k_mono",
        lambda path: torch.zeros(60 * 16000, dtype=torch.int16),
    )
    monkeypatch.setattr(
        gigaam_engine,
        "speech_intervals",
        lambda wav, vad, *, threshold: [(0.0, 18.0), (19.0, 37.0), (38.0, 56.0)],
    )

    class _CancelAfter:
        """cancel_check, возвращающий True начиная с (after+1)-го вызова."""

        def __init__(self, after: int) -> None:
            self.calls = 0
            self.after = after

        def __call__(self) -> bool:
            self.calls += 1
            return self.calls > self.after

    with pytest.raises(InferenceCancelledError):
        eng._transcribe_longform("x.wav", word_timestamps=False, cancel_check=_CancelAfter(1))

    assert model.forward_calls == 1  # успел только первый батч


# ---------------------------------------------------- iter_segments (стриминг)


class _CancelAfter:
    """cancel_check, возвращающий True начиная с (after+1)-го вызова."""

    def __init__(self, after: int) -> None:
        self.calls = 0
        self.after = after

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.after


def _mock_longform_audio(
    monkeypatch: pytest.MonkeyPatch, intervals: list[tuple[float, float]]
) -> None:
    monkeypatch.setattr(
        gigaam_engine,
        "decode_to_int16_16k_mono",
        lambda path: torch.zeros(60 * 16000, dtype=torch.int16),
    )
    monkeypatch.setattr(gigaam_engine, "speech_intervals", lambda wav, vad, *, threshold: intervals)


def test_iter_segments_yields_each_longform_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(MODEL="v3_ctc", BATCH_SIZE=4)
    model = _FakeASRModel([("раз", None), ("два", None), ("три", None)])
    eng = _bare_engine(settings, model)
    monkeypatch.setattr(gigaam_engine, "probe_duration", lambda path: 60.0)
    _mock_longform_audio(monkeypatch, [(0.0, 18.0), (19.0, 37.0), (38.0, 56.0)])

    segs = list(eng.iter_segments("x.wav", word_timestamps=False))

    assert [s.text for s in segs] == ["раз", "два", "три"]
    assert [(s.start, s.end) for s in segs] == [(0.0, 18.0), (19.0, 37.0), (38.0, 56.0)]


def test_iter_segments_short_yields_single_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _bare_engine(Settings(MODEL="v3_ctc"), _FakeASRModel([]))
    monkeypatch.setattr(gigaam_engine, "probe_duration", lambda path: 10.0)
    one = ASRResult("привет", 10.0, [SegmentTS(text="привет", start=0.0, end=10.0)])
    monkeypatch.setattr(eng, "_transcribe_short", _recorder([], "short", one))

    segs = list(eng.iter_segments("x.wav", word_timestamps=False))

    assert [s.text for s in segs] == ["привет"]


def test_iter_segments_cancels_between_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(MODEL="v3_ctc", BATCH_SIZE=1)
    model = _FakeASRModel([("раз", None)])  # хватит на один обработанный батч
    eng = _bare_engine(settings, model)
    monkeypatch.setattr(gigaam_engine, "probe_duration", lambda path: 60.0)
    _mock_longform_audio(monkeypatch, [(0.0, 18.0), (19.0, 37.0), (38.0, 56.0)])

    got: list[SegmentTS] = []
    with pytest.raises(InferenceCancelledError):
        for seg in eng.iter_segments("x.wav", word_timestamps=False, cancel_check=_CancelAfter(1)):
            got.append(seg)

    assert [s.text for s in got] == ["раз"]  # первый чанк выдан до отмены
    assert model.forward_calls == 1


def test_iter_segments_raises_when_exceeding_max_audio_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _bare_engine(Settings(MODEL="v3_ctc", MAX_AUDIO_SECONDS=20), _FakeASRModel([]))
    monkeypatch.setattr(gigaam_engine, "probe_duration", lambda path: 30.0)

    with pytest.raises(AudioTooLongError):
        list(eng.iter_segments("x.wav", word_timestamps=False))


def test_transcribe_too_long_valueerror_falls_back_to_longform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ModelTooLong:
        def transcribe(self, wav_path: str, word_timestamps: bool) -> object:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")

    eng = _bare_engine(Settings(MODEL="v3_ctc"), _ModelTooLong())
    monkeypatch.setattr(gigaam_engine, "probe_duration", lambda path: 24.0)  # short-путь
    calls: list[str] = []
    monkeypatch.setattr(
        eng, "_transcribe_longform", _recorder(calls, "long", ASRResult("", 24.0, []))
    )

    eng.transcribe("x.wav", word_timestamps=False)
    assert calls == ["long"]
