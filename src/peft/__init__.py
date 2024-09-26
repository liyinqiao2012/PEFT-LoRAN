__version__ = "0.4.0"

from .auto import (
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForTokenClassification,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForFeatureExtraction,
)
from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING, get_peft_config, get_peft_model
from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
    PeftModelForQuestionAnswering,
    PeftModelForFeatureExtraction,
)
from .tuners import (
    AdaptionPromptConfig,
    AdaptionPromptModel,
    LoraConfig,
    LoraModel,
    MLoraConfig,
    MLoraModel,
    IA3Config,
    IA3Model,
    AdaLoraConfig,
    AdaLoraModel,
    PrefixEncoder,
    PrefixTuningConfig,
    PromptEmbedding,
    PromptEncoder,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptTuningConfig,
    PromptTuningInit,
)
from .utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    PeftConfig,
    PeftType,
    PromptLearningConfig,
    TaskType,
    bloom_model_postprocess_past_key_value,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    shift_tokens_right,
)
