
from dify_plugin import LargeLanguageModel
from dify_plugin.entities.model import AIModelEntity


MODEL_CLASS_MAP = {}


class ZenMuxLargeLanguageModel(LargeLanguageModel):
    """
    Model class for zenmux large language model.
    """

    def __init__(self, model_schemas: list[AIModelEntity]) -> None:
        super().__init__(model_schemas)

        model_class_map = {}
        for model_schema in model_schemas:
            model_class = MODEL_CLASS_MAP[model_schema.model]
            if model_class not in model_class_map:
                model_schema_list = model_class_map.setdefault(model_class, [])
            else:
                model_schema_list = model_class_map[model_class]
            model_schema_list.append(model_schema)

        default_model_class = MODEL_CLASS_MAP["*"]

        model_map = {}
        for model_class in model_class_map:
            model_schema_list = model_class_map[model_class]
            model = model_class(model_schema_list)
            if model_class == default_model_class:
                self.default_model = model

            for model_schema in model_schema_list:
                model_map[model_schema.model] = model

        self.model_map = model_map

    def validate_credentials(self, model: str, *args, **kwargs):
        model_obj = self.model_map.get(model, self.default_model)
        return model_obj.validate_credentials(model, *args, **kwargs)

    @property
    def _invoke_error_mapping(self):
        return self.default_model._invoke_error_mapping

    def _invoke(self, model: str, *args, **kwargs):
        model_obj = self.model_map.get(model, self.default_model)
        return model_obj._invoke(model, *args, **kwargs)

    def get_num_tokens(self, model: str, *args, **kwargs):
        model_obj = self.model_map.get(model, self.default_model)
        return model_obj.get_num_tokens(model, *args, **kwargs)
