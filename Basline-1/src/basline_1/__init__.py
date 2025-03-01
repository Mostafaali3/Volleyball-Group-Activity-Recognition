import torch 
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class GroupActivity():
    def __init__(self,line:str, video_number):
        self.text = line.split()
        self.main_frame_id = self.text[0]
        self.activity_class = self.text[1]
        self.video_number = video_number
        
        if self.main_frame_id:
            self.main_frame_number = int(self.main_frame_id.split('.')[0])
            self.frames_ids = [x for x in range(self.main_frame_number-5, self.main_frame_number+5)]
            
    def get_frame_ids(self):
        print(self.frames_ids)
        print(self.activity_class)
        print(self.video_number)
        
        
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_path = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(self.image_path[index]), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

            
def get_frames_from_id(line:GroupActivity, dataset_path = '/kaggle/input/volleyball/volleyball_/videos'):
    frames = [f'{dataset_path}/{line.video_number}/{line.main_frame_number}/{x}.jpg' for x in line.frames_ids]
    labels = [line.activity_class for _ in range(len(frames))]
    return frames, labels
    
def get_dataloaders_from_dataset(dataset_path = '/kaggle/input/volleyball/volleyball_/videos', train_video_numbers = [], test_video_numbers = [], valid_video_numbers = [], transform = None):
    train_data, test_data, valid_data = [], [], []
    train_labels, test_labels, valid_labels = [], [], []
    for i in range(55):
        with open(f"{dataset_path}/{i}/annotations.txt", "r") as file:
            for line in file:
                # print(line.strip())
                group_activity = GroupActivity(line.strip(), i)
                frames, labels = get_frames_from_id(group_activity, dataset_path)
                if group_activity.video_number in train_video_numbers:
                    train_data = train_data + frames
                    train_labels = train_labels + labels
                elif group_activity.video_number in valid_video_numbers:
                    valid_data += frames
                    valid_labels += labels
                elif group_activity.video_number in test_video_numbers:
                    test_data += frames
                    test_labels += labels
    train_dataset = CustomDataset(train_data, train_labels,transform = transform)
    test_dataset = CustomDataset(test_data, test_labels, transform = transform)
    valid_dataset = CustomDataset(valid_data, valid_labels, transform = transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    
    return train_loader, test_loader, valid_loader
                
                

    
# loop on folders inside videos_ (0,1,2,...)
# get the annotations file 
# loop on every line and put in instances from GroupActivity
# get into the folder with same name of the main frame id 
# call the function to get the 10 frames and their labels 
   
train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
valid_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4 ,5, 9 ,11 ,14 ,20 ,21 ,25 ,29 ,34 ,35 ,37 ,43 ,44 ,45 ,47]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_loader, test_loader, valid_loader = get_dataloaders_from_dataset(train_video_numbers = train_videos, test_video_numbers = test_videos, valid_video_numbers = valid_videos, transform = transform)