import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import gruut
import librosa
import numpy as np
import pysbd
import torch

from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.vocoder.models.base_vocoder import BaseVocoder
from TTS.vocoder.configs.shared_configs import BaseVocoderConfig
from TTS.tts.utils.speakers import SpeakerManager

# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
from TTS.tts.utils.synthesis import synthesis, trim_silence
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.vocoder.utils.generic_utils import interpolate_vocoder_input


@dataclass
class VoiceConfig:
    tts_checkpoint: str
    tts_config_path: str
    name: str = ""
    lang: str = ""
    tts_speakers_file: str = ""
    vocoder_checkpoint: str = ""
    vocoder_config_path: str = ""
    encoder_checkpoint: str = ""
    encoder_config_path: str = ""
    use_cuda: bool = False

    tts_model: Optional[BaseTTS] = None
    tts_config: Optional[BaseTTSConfig] = None
    ap: Optional[AudioProcessor] = None
    output_sample_rate: int = 0
    use_phonemes: bool = False

    vocoder_model: Optional[BaseVocoder] = None
    vocoder_config: Optional[BaseVocoderConfig] = None
    vocoder_ap: Optional[AudioProcessor] = None

    num_speakers: int = 0
    d_vector_dim: int = 0

    use_multi_speaker: bool = False
    speaker_manager: Optional[SpeakerManager] = None
    use_gst: bool = False

    def load(self) -> None:
        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."

        self._load_tts()
        self.output_sample_rate = self.tts_config.audio["sample_rate"]
        if self.vocoder_checkpoint:
            self._load_vocoder()
            self.output_sample_rate = self.vocoder_config.audio["sample_rate"]

    def _load_tts(self) -> None:
        """Load the TTS model."""
        self.tts_config = load_config(self.tts_config_path)
        self.use_phonemes = self.tts_config.use_phonemes
        if not self.lang:
            self.lang = self.tts_config.phoneme_language or "en"

        self.ap = AudioProcessor(verbose=False, **self.tts_config.audio)

        self.tts_model = setup_tts_model(config=self.tts_config)
        self.tts_model.load_checkpoint(self.tts_config, self.tts_checkpoint, eval=True)
        if self.use_cuda:
            self.tts_model.cuda()
        self._set_tts_speaker_file()

        self.use_multi_speaker = hasattr(self.tts_model, "speaker_manager") and self.tts_model.num_speakers > 1
        self.speaker_manager = getattr(self.tts_model, "speaker_manager", None)
        # TODO: set this from SpeakerManager
        self.use_gst = self.tts_config.get("use_gst", False)

    def _load_vocoder(self) -> None:
        """Load the vocoder model."""
        self.vocoder_config = load_config(self.vocoder_config_path)
        self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config.audio)
        self.vocoder_model = setup_vocoder_model(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, self.vocoder_checkpoint, eval=True)
        if self.use_cuda:
            self.vocoder_model.cuda()

    def _set_tts_speaker_file(self):
        """Set the TTS speaker file used by a multi-speaker model."""
        # setup if multi-speaker settings are in the global model config
        if hasattr(self.tts_config, "use_speaker_embedding") and self.tts_config.use_speaker_embedding is True:
            if self.tts_config.use_d_vector_file:
                self.tts_speakers_file = (
                    self.tts_speakers_file if self.tts_speakers_file else self.tts_config["d_vector_file"]
                )
                self.tts_config["d_vector_file"] = self.tts_speakers_file
            else:
                self.tts_speakers_file = (
                    self.tts_speakers_file if self.tts_speakers_file else self.tts_config["speakers_file"]
                )

        # setup if multi-speaker settings are in the model args config
        if (
            (not self.tts_speakers_file)
            and hasattr(self.tts_config, "model_args")
            and hasattr(self.tts_config.model_args, "use_speaker_embedding")
            and self.tts_config.model_args.use_speaker_embedding
        ):

            _args = self.tts_config.model_args
            if _args.use_d_vector_file:
                self.tts_speakers_file = self.tts_speakers_file if self.tts_speakers_file else _args["d_vector_file"]
                _args["d_vector_file"] = self.tts_speakers_file
            else:
                self.tts_speakers_file = self.tts_speakers_file if self.tts_speakers_file else _args["speakers_file"]

    def get_speaker_embedding(self, speaker_idx: str = "", speaker_wav=None):
        speaker_embedding = None
        speaker_id = None
        if self.tts_speakers_file:
            if speaker_idx and isinstance(speaker_idx, str):
                if self.tts_config.use_d_vector_file:
                    # get the speaker embedding from the saved d_vectors.
                    speaker_embedding = self.tts_model.speaker_manager.get_d_vectors_by_speaker(speaker_idx)[0]
                else:
                    # get speaker idx from the speaker name
                    speaker_id = self.tts_model.speaker_manager.speaker_ids[speaker_idx]

            elif not speaker_idx and not speaker_wav:
                raise ValueError(
                    " [!] Look like you use a multi-speaker model. "
                    "You need to define either a `speaker_idx` or a `style_wav` to use a multi-speaker model."
                )
            else:
                speaker_embedding = None
        else:
            if speaker_idx:
                raise ValueError(
                    f" [!] Missing speaker.json file path for selecting speaker {speaker_idx}."
                    "Define path for speaker.json if it is a multi-speaker model or remove defined speaker idx. "
                )

        return speaker_id, speaker_embedding


class Synthesizer(object):
    def __init__(
        self,
        tts_checkpoint: str,
        tts_config_path: str,
        tts_speakers_file: str = "",
        vocoder_checkpoint: str = "",
        vocoder_config: str = "",
        encoder_checkpoint: str = "",
        encoder_config: str = "",
        use_cuda: bool = False,
        tts_name: str = "",
        extra_voices: Optional[Iterable[VoiceConfig]] = None,
    ) -> None:
        """General 🐸 TTS interface for inference. It takes a tts and a vocoder
        model and synthesize speech from the provided text.

        The text is divided into a list of sentences using `pysbd` and synthesize
        speech on each sentence separately.

        If you have certain special characters in your text, you need to handle
        them before providing the text to Synthesizer.

        TODO: set the segmenter based on the source language

        Args:
            tts_checkpoint (str): path to the tts model file.
            tts_config_path (str): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            encoder_checkpoint (str, optional): path to the speaker encoder model file. Defaults to `""`,
            encoder_config (str, optional): path to the speaker encoder config file. Defaults to `""`,
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.tts_speakers_file = tts_speakers_file
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.encoder_checkpoint = encoder_checkpoint
        self.encoder_config = encoder_config
        self.use_cuda = use_cuda

        self.voices: List[VoiceConfig] = [
            VoiceConfig(
                tts_checkpoint=tts_checkpoint,
                tts_config_path=tts_config_path,
                vocoder_checkpoint=vocoder_checkpoint,
                vocoder_config_path=vocoder_config,
                encoder_checkpoint=encoder_checkpoint,
                encoder_config_path=encoder_config,
                use_cuda=use_cuda,
                name=tts_name,
            )
        ] + list(extra_voices)

        assert self.voices, "At least one voice is required"

        # name -> VoiceConfig
        # name is something like "tts_models/en/ljspeech/tacotron2-DDC"
        self.voice_by_name: Dict[str, VoiceConfig] = {}

        # lang -> VoiceConfig
        self.voice_by_lang: Dict[str, VoiceConfig] = {}

        # lang -> Segmenter
        self.seg_by_lang: Dict[str, pysbd.Segmenter] = {}

        # Load voices and segmenters
        for voice in self.voices:
            voice.load()

            if voice.name not in self.voice_by_name:
                # Cache voice by name (first gets priority)
                self.voice_by_name[voice.name] = voice

            if voice.lang not in self.voice_by_lang:
                # Cache voice by language (first gets priority)
                self.voice_by_lang[voice.lang] = voice

            if len(voice.lang) > 2:
                short_lang = voice.lang[:2]
                if short_lang not in self.voice_by_lang:
                    # Cache voice by 2-character language code (first gets priority)
                    self.voice_by_lang[short_lang] = voice

            if voice.lang not in self.seg_by_lang:
                # Load sentence segementer for language
                self.seg_by_lang[voice.lang] = self._get_segmenter(voice.lang[:2])

        # Set properties for default voice
        self.default_voice = self.voices[0]
        self.default_lang = self.default_voice.lang or "en"

        self.tts_model = self.default_voice.tts_model
        self.tts_config = self.default_voice.tts_config
        self.vocoder_model = self.default_voice.vocoder_model
        self.vocoder_config = self.default_voice.vocoder_config
        self.ap = self.default_voice.vocoder_ap or self.default_voice.ap
        self.output_sample_rate = self.ap.sample_rate
        self.speaker_manager = self.default_voice.speaker_manager
        self.num_speakers = self.default_voice.num_speakers
        self.tts_speakers = {}
        self.d_vector_dim = self.default_voice.d_vector_dim
        self.seg = self.seg_by_lang[self.default_lang[:2]]

    @staticmethod
    def _get_segmenter(lang: str):
        """get the sentence segmenter for the given language.

        Args:
            lang (str): target language code.

        Returns:
            [type]: [description]
        """
        return pysbd.Segmenter(language=lang, clean=True)

    def split_into_sentences(self, text, lang: str = "") -> List[str]:
        """Split give text into sentences.

        Args:
            text (str): input text in string format.
            lang (str): language of input text (empty for default language)

        Returns:
            List[str]: list of sentences.
        """
        lang = lang or self.default_lang
        return self.seg_by_lang.get(lang, self.seg).segment(text)

    def save_wav(self, wav: List[int], path: str) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
        """
        wav = np.array(wav)
        self.ap.save_wav(wav, path, self.output_sample_rate)

    def tts(
        self,
        text: str,
        speaker_idx: str = "",
        speaker_wav=None,
        style_wav=None,
        voice_name: str = "",
        lang: str = "",
        ssml: bool = False,
    ) -> List[int]:
        """🐸 TTS magic. Run all the models and generate speech.

        Args:
            text (str): input text.
            speaker_idx (str, optional): spekaer id for multi-speaker models. Defaults to "".
            speaker_wav ():
            style_wav ([type], optional): style waveform for GST. Defaults to None.

        Returns:
            List[int]: [description]
        """

        # Set default voice and language
        voice = self.default_voice
        if voice_name:
            voice = self.voice_by_name.get(voice_name, voice)
        elif lang:
            voice = self.voice_by_lang.get(lang, voice)

        lang = voice.lang

        start_time = time.time()
        wavs = []

        if ssml:
            # Split SSML into sentences using gruut.
            #
            # POS tagging and phonemization is disabled since it will be done in
            # a later stage.
            #
            # Numbers are not verbalized here either in order to not interfere
            # with the text cleaners.
            sens = list(
                gruut.sentences(
                    text,
                    lang=lang,
                    ssml=True,
                    pos=False,
                    phonemize=False,
                    verbalize_numbers=False,
                )
            )
            print(" > Text splitted to sentences.")
            print([(s.text, s.voice, s.lang) for s in sens])
        else:
            sens = self.split_into_sentences(text, lang=lang)
            print(" > Text splitted to sentences.")
            print(sens)

        # Process each sentence, which may be a string or a gruut.Sentence
        # object.
        for sen in sens:
            text = sen
            sen_voice = voice
            sen_speaker_idx = speaker_idx
            sen_speaker_wav = speaker_wav

            if isinstance(sen, gruut.Sentence):
                # Input text was SSML, so we need to determine which TTS model
                # to use for synthesis.
                text = sen.text_with_ws
                maybe_sen_voice: Optional[VoiceConfig] = None
                maybe_sen_speaker_idx = None

                if sen.voice:
                    # voice was specified for this sentence
                    sen_voice_name = sen.voice
                    if "#" in sen_voice_name:
                        # voice may have format "<name>#<speaker_idx>" for multispeaker models
                        sen_voice_name, maybe_sen_speaker_idx = sen_voice_name.split("#", maxsplit=1)

                    maybe_sen_voice = self.voice_by_name.get(sen_voice_name)
                    if maybe_sen_voice is None:
                        # Voice was not found by name.
                        # Reset to default voice and speaker idx.
                        maybe_sen_speaker_idx = None

                if (maybe_sen_voice is None) and sen.lang:
                    # lang was specified for this sentence
                    maybe_sen_voice = self.voice_by_lang.get(sen.lang)

                if maybe_sen_voice is not None:
                    # voice or lang was successfully used to locate a TTS model
                    sen_voice = maybe_sen_voice
                    sen_speaker_idx = maybe_sen_speaker_idx
                    sen_speaker_wav = None

            use_gl = sen_voice.vocoder_model is None

            # handle multi-speaker
            speaker_id, speaker_embedding = sen_voice.get_speaker_embedding(
                sen_speaker_idx, speaker_wav=sen_speaker_wav
            )

            # synthesize voice
            outputs = synthesis(
                model=sen_voice.tts_model,
                text=text,
                CONFIG=sen_voice.tts_config,
                use_cuda=sen_voice.use_cuda,
                ap=sen_voice.ap,
                speaker_id=speaker_id,
                style_wav=style_wav,
                enable_eos_bos_chars=sen_voice.tts_config.enable_eos_bos_chars,
                use_griffin_lim=use_gl,
                d_vector=speaker_embedding,
            )
            waveform = outputs["wav"]
            waveform_sample_rate = sen_voice.ap.sample_rate
            mel_postnet_spec = outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()

            if not use_gl:
                # denormalize tts output based on tts audio config
                mel_postnet_spec = sen_voice.ap.denormalize(mel_postnet_spec.T).T
                device_type = "cuda" if sen_voice.use_cuda else "cpu"
                # renormalize spectrogram based on vocoder config
                vocoder_input = sen_voice.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    sen_voice.vocoder_config["audio"]["sample_rate"] / sen_voice.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = sen_voice.vocoder_model.inference(vocoder_input.to(device_type))
                waveform_sample_rate = sen_voice.vocoder_ap.sample_rate

            if sen_voice.use_cuda and not use_gl:
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            waveform = waveform.squeeze()

            # trim silence
            waveform = trim_silence(waveform, sen_voice.ap)

            if waveform_sample_rate != self.output_sample_rate:
                # Resample to sample rate of default voice
                waveform = librosa.resample(waveform, orig_sr=waveform_sample_rate, target_sr=self.output_sample_rate)

            wavs += list(waveform)
            wavs += [0] * 10000

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / voice.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wavs
