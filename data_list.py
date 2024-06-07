<<<<<<< HEAD

=======
>>>>>>> 3b137b1 (Initial commit)
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


<<<<<<< HEAD


def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def get_classes(image_list):
    class_dict1={}
    class_dict2={}
    for index in range(len(image_list)):
        path, target = image_list[index]
        classes=path.split('/')[-2]
        if target not in class_dict1.keys():
            class_dict1[target]=classes
        if classes not in class_dict2.keys():
            class_dict2[classes]=target
        
    return class_dict1,class_dict2


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


        



class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,class_list=[]):
        imgs = make_dataset(image_list, labels)
        self.class_dict1,self.class_dict2=get_classes(imgs)
        
        self.classes=class_list
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders"))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
       
        self.loader = rgb_loader
       
    def __getitem__(self, index):
        path, target = self.imgs[index]
        
        img = self.loader(path)
       
=======
def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [
                (val.split()[0], np.array([int(la) for la in val.split()[1:]]))
                for val in image_list
            ]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def get_classes(image_list):
    class_dict1 = {}
    class_dict2 = {}
    for index in range(len(image_list)):
        path, target = image_list[index]
        classes = path.split("/")[-2]
        if target not in class_dict1.keys():
            class_dict1[target] = classes
        if classes not in class_dict2.keys():
            class_dict2[classes] = target

    return class_dict1, class_dict2


def rgb_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class ImageList(Dataset):
    def __init__(
        self,
        image_list,
        labels=None,
        transform=None,
        target_transform=None,
        class_list=[],
    ):
        imgs = make_dataset(image_list, labels)
        self.class_dict1, self.class_dict2 = get_classes(imgs)

        self.classes = class_list
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders"))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

        self.loader = rgb_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]

        img = self.loader(path)

>>>>>>> 3b137b1 (Initial commit)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

<<<<<<< HEAD
        return img, target,'a'
        
    def __len__(self):
        return len(self.imgs)
    




class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform1=None, transform2=None,mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
=======
        return img, target, "a"

    def __len__(self):
        return len(self.imgs)


class ImageList_idx(Dataset):
    def __init__(
        self, image_list, labels=None, transform1=None, transform2=None, mode="RGB"
    ):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )
>>>>>>> 3b137b1 (Initial commit)

        self.imgs = imgs
        self.transform1 = transform1
        self.transform2 = transform2

        self.loader = rgb_loader
<<<<<<< HEAD
        
=======

>>>>>>> 3b137b1 (Initial commit)
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform1 is not None:
            img1 = self.transform1(img)

        if self.transform2 is not None:
<<<<<<< HEAD
            img2=self.transform2(img)
        else:
            img2=0
     
        return [img1,img2], target, index
        
=======
            img2 = self.transform2(img)
        else:
            img2 = 0

        return [img1, img2], target, index
>>>>>>> 3b137b1 (Initial commit)

    def __len__(self):
        return len(self.imgs)


<<<<<<< HEAD


class ImageList_idx_fast(Dataset):
    def __init__(self, image_list, labels=None, transform1=None, transform2=None,mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
=======
class ImageList_idx_fast(Dataset):
    def __init__(
        self, image_list, labels=None, transform1=None, transform2=None, mode="RGB"
    ):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )
>>>>>>> 3b137b1 (Initial commit)

        self.imgs = imgs
        self.transform1 = transform1
        self.transform2 = transform2
<<<<<<< HEAD
     
       
   
        self.loader = rgb_loader
    

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img1=0
        img2=0
        return [img1,img2], target, index
        
=======

        self.loader = rgb_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img1 = 0
        img2 = 0
        return [img1, img2], target, index
>>>>>>> 3b137b1 (Initial commit)

    def __len__(self):
        return len(self.imgs)


<<<<<<< HEAD




class ImageList_idx1(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, transform1=None,mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
=======
class ImageList_idx1(Dataset):
    def __init__(
        self,
        image_list,
        labels=None,
        transform=None,
        target_transform=None,
        transform1=None,
        mode="RGB",
    ):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )
>>>>>>> 3b137b1 (Initial commit)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
<<<<<<< HEAD
        self.transform1=transform1

        self.loader = rgb_loader
     
=======
        self.transform1 = transform1

        self.loader = rgb_loader
>>>>>>> 3b137b1 (Initial commit)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img1 = self.transform(img)
        if self.transform1 is not None:
<<<<<<< HEAD
            img2=self.transform1(img)
=======
            img2 = self.transform1(img)
>>>>>>> 3b137b1 (Initial commit)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform1 is not None:
<<<<<<< HEAD
            return [img1,img2], target, index
=======
            return [img1, img2], target, index
>>>>>>> 3b137b1 (Initial commit)
        else:
            return img1, target, index

    def __len__(self):
        return len(self.imgs)


<<<<<<< HEAD

class Splited_List(Dataset):
    def __init__(self,all_fea,all_output,all_pse,d1):

        self.all_fea = all_fea
        self.all_output = all_output
        self.all_pse=all_pse
        self.d1 = d1
    def __getitem__(self, index):
        if index in self.d1:
            domain_label=0
        else:
            domain_label=1 
        return self.all_fea[index],domain_label#,self.all_output[index],self.all_pse[index],domain_label
=======
class Splited_List(Dataset):
    def __init__(self, all_fea, all_output, all_pse, d1):

        self.all_fea = all_fea
        self.all_output = all_output
        self.all_pse = all_pse
        self.d1 = d1

    def __getitem__(self, index):
        if index in self.d1:
            domain_label = 0
        else:
            domain_label = 1
        return (
            self.all_fea[index],
            domain_label,
        )  # ,self.all_output[index],self.all_pse[index],domain_label
>>>>>>> 3b137b1 (Initial commit)

    def __len__(self):
        return self.all_fea.shape[0]
