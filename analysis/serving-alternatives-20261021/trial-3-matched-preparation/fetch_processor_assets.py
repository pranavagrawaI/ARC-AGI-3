"""Read only small metadata assets from pinned Kaggle model versions. No weights.

Run with the installed Kaggle Python. Underlying request.path supports a single
file although KaggleApi.model_instance_version_download does not expose it.
"""
import hashlib
import json
import signal
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.models.types.model_api_service import ApiDownloadModelInstanceVersionRequest, ModelFramework

HERE=Path(__file__).resolve().parent
MODELS={
    "candidate":"lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1",
    "control":"dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
}
ALLOWED={"config.json","preprocessor_config.json","processor_config.json","tokenizer_config.json",
         "tokenizer.json","special_tokens_map.json","chat_template.jinja","generation_config.json"}


def main():
    signal.alarm(240)
    api=KaggleApi();api.authenticate()
    receipt={"weights_downloaded":False,"gpu_used":False,"total_bytes":0,"models":{}}
    destination=HERE/"processor-assets";destination.mkdir(exist_ok=True)
    try:
        for label, model in MODELS.items():
            token=None;files=[]
            for _ in range(8):
                result=api.model_instance_version_files(model,page_token=token,page_size=100)
                row=result.to_dict();files.extend(row["files"]);token=row.get("nextPageToken")
                if not token:break
            if token:raise RuntimeError("model metadata listing exceeds bounded pages")
            directory=destination/label;directory.mkdir(exist_ok=True)
            info={"model_version":model,"files":files,"assets":{}}
            receipt["models"][label]=info
            for file in files:
                name=file["name"]
                if name not in ALLOWED:continue
                expected=int(file["size"])
                if expected>32*1024**2:raise RuntimeError("metadata file over 32MiB cap: "+name)
                existing=directory/name
                if existing.exists() and existing.stat().st_size==expected:
                    blob=existing.read_bytes()
                    receipt["total_bytes"]+=len(blob)
                    info["assets"][name]={"bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest(),"existing_local":True}
                    continue
                owner,slug,framework,instance,version=model.split("/")
                request=ApiDownloadModelInstanceVersionRequest()
                request.owner_slug=owner;request.model_slug=slug
                request.framework=api.lookup_enum(ModelFramework,ModelFramework.MODEL_FRAMEWORK_API,framework)
                request.instance_slug=instance;request.version_number=int(version);request.path=name
                with api.build_kaggle_client() as client:
                    response=client.models.model_api_client.download_model_instance_version(request)
                    try:
                        response.raise_for_status()
                        blob=bytearray()
                        for chunk in response.iter_content(chunk_size=65536):
                            blob.extend(chunk)
                            if len(blob)>32*1024**2 or receipt["total_bytes"]+len(blob)>80*1024**2:
                                raise RuntimeError("metadata byte cap exceeded")
                    finally:response.close()
                if len(blob)!=expected:raise RuntimeError("metadata size mismatch: "+name)
                (directory/name).write_bytes(blob)
                receipt["total_bytes"]+=len(blob)
                info["assets"][name]={"bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest()}
                print(label,name,len(blob),flush=True)
        receipt["complete"]=True
    finally:
        (destination/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
        signal.alarm(0)


if __name__=="__main__":main()
