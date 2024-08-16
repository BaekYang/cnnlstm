import os
from PIL import Image
import torch
import params
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 하위 폴더와 레이블 매핑
        class_map = {
            "0-standing": 0,
            "1-ready": 1,
            "2-aiming": 2
        }

        # 모든 하위 폴더 순회하며 이미지 경로와 레이블 추출
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for class_name, label in class_map.items():
                    class_folder = os.path.join(subdir_path, class_name)
                    if os.path.exists(class_folder):
                        for img_file in os.listdir(class_folder):
                            img_path = os.path.join(class_folder, img_file)
                            self.image_paths.append(img_path)
                            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    dataset = CustomDataset(root_dir=params.train_data_path, transform=transform)
    
    # 데이터셋 크기 확인
    print("Dataset size:", len(dataset))
    
    # 첫 번째 데이터 샘플 로드
    image, label = dataset[0]
    print("First image size:", image.size())
    print("Label:", label)    
