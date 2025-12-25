from typing import Any, Type
from pydantic import BaseModel
from jsonschema import validate as js_validate, ValidationError as JSValidationError


def jsonschema_to_pydantic_model(name: str, schema: dict) -> Type[BaseModel]:
    """
    Returns a BaseModel subclass that:
      - validates data against the provided JSON Schema
      - reports the same JSON Schema via model_json_schema()
    """
    class _SchemaModel(BaseModel):
        @classmethod
        def model_json_schema(cls, *args, **kwargs) -> dict:
            # Give back the original JSON schema exactly
            return schema

        @classmethod
        def model_validate(cls, obj: Any, *args, **kwargs):
            # Hard validation against JSON Schema
            try:
                js_validate(instance=obj, schema=schema)
            except JSValidationError as e:
                raise ValueError(f"JSON Schema validation failed: {e.message}") from e
            # If it passes schema validation, just return the raw object
            # wrapped in a trivial BaseModel instance.
            inst = cls.model_construct()
            inst.__dict__["_value"] = obj
            return inst

        def model_dump(self, *args, **kwargs):
            return self.__dict__.get("_value")

    _SchemaModel.__name__ = name
    return _SchemaModel
