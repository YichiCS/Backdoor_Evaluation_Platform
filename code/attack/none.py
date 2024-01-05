import torch


class poison_generator():

    def __init__(self, img_size, dataset, path):
        
        self.img_size = img_size
        self.dataset = dataset
        self.path = path

        self.num_img = len(dataset)

    def generate_poisoned_set(self):

        img_set = []
        label_set = []

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, [], label_set
    
class poison_transform():
    def __init__(self):
        pass

    def transform(self, data, labels):
        return data, labels
