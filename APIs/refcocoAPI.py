from typing import List, Annotated, Optional
from pydantic import BaseModel, Field
import os
from .refer_python3_2024.refer import REFER



DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "refer_python3_2024",
    "data",
)


class BboxModel(BaseModel):
    x : float = Field(description="bbox 좌측 상단 점의 x좌표")
    y : float = Field(description="bbox 좌측 상단 점의 y좌표")
    w : float = Field(description="bbox width")
    h : float = Field(description="bbox height")


class ImageSizeModel(BaseModel):
    width : float = Field(description="이미지 폭")
    height : float = Field(description="이미지 높이")


class RefcocoModel(BaseModel):
    image_id : str = Field(description="mscoco 이미지 id")
    image_path : Optional[str] = Field(default=None, description="이미지가 저장된 경로")
    size : Optional[ImageSizeModel] = Field(default=None, description="이미지 사이즈, (w, h)")
    bbox : BboxModel = Field(description="bbox좌표, (x, y, w, h), center based bbox 아님")
    annotation : str = Field(description="bbox가 가르키는 대상에 대한 참조 표현")


RefcocoModel_list = Annotated[List[RefcocoModel], Field(description="Refcoco 데이터 모델 리스트 입니다")]


def _candidate_image_roots(data_root: str, refer: REFER) -> list[str]:
    candidates = [
        refer.IMAGE_DIR,
        os.path.join(data_root, "images", "mscoco", "images", "train2014"),
        os.path.join(data_root, "train2014"),
        os.path.join(data_root, "images", "train2014"),
    ]

    unique_candidates: list[str] = []
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in unique_candidates:
            unique_candidates.append(normalized)
    return unique_candidates


def _resolve_image_path(file_name: str, data_root: str, refer: REFER) -> Optional[str]:
    for root in _candidate_image_roots(data_root=data_root, refer=refer):
        candidate = os.path.join(root, file_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_refcoco(
    data_root=DEFAULT_DATA_ROOT,
    dataset="refcoco",
    splitby="unc",
    split="train",
    include_image_metadata=False,
) -> RefcocoModel_list:
    refer = REFER(data_root=data_root, dataset=dataset, splitBy=splitby)
    ref_ids = refer.getRefIds(split=split)
    refs = refer.loadRefs(ref_ids)

    annotations = [list(set(annt["sent"] for annt in ref["sentences"])) for ref in refs]
    print("num of annotations : ", len(annotations))

    bboxes = [refer.getRefBox(ref_id) for ref_id in ref_ids]
    print("num of bboxes : ", len(bboxes))

    image_ids = [str(ref["image_id"]) for ref in refs]
    image_paths = [None] * len(refs)
    sizes = [None] * len(refs)

    if include_image_metadata:
        images = refer.loadImgs([ref["image_id"] for ref in refs])
        file_names = [image["file_name"] for image in images]
        sizes = [
            ImageSizeModel(width=image["width"], height=image["height"])
            for image in images
        ]
        image_paths = [
            _resolve_image_path(file_name=file_name, data_root=data_root, refer=refer)
            for file_name in file_names
        ]

        print("num of file_names : ", len(file_names))
        print("num of sizes : ", len(sizes))

    train_data = []

    for i in range(len(annotations)):
        bbox = BboxModel(
            x=bboxes[i][0],
            y=bboxes[i][1],
            w=bboxes[i][2],
            h=bboxes[i][3],
        )
        for annt in annotations[i]:
            train_data.append(
                RefcocoModel(
                    image_id=image_ids[i],
                    image_path=image_paths[i],
                    bbox=bbox,
                    size=sizes[i],
                    annotation=annt,
                )
            )

    return train_data


if __name__ == "__main__":
    try:
        train_data = load_refcoco(
            dataset="refcoco",
            split="train",
            splitby="unc",
            include_image_metadata=True,
        )
        print("loaded annotation entries :", len(train_data))
        
        
        for i in range(3):
            temp = train_data[i]
            print("====================================", end="\n")
            print(f"Image Path : {temp.image_id}")
            print(f"Image Size : {temp.size.model_dump()}")
            print(f"Bbox : {temp.bbox.model_dump()}")
            print(f"Annotation : {temp.annotation}")


    except FileNotFoundError as exc:
        print(f"missing dataset file: {exc}")
