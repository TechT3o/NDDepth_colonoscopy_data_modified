import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import DistributedSamplerNoEvenlyDivisible
import cv2
import random
import copy


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    print("Preprocessing Function")
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == "online_eval":
            with open(args.filenames_file_eval, "r") as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, "r") as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = 518.8579  # Update if needed

        # rgb_file = sample_path.split()[0]
        # depth_file = sample_path.split()[1]
        #
        # image_path = os.path.join(self.args.data_path, rgb_file)
        # depth_path = os.path.join(self.args.gt_path, depth_file)
        #
        # image = Image.open(image_path).convert("RGB")
        # depth_gt = Image.open(depth_path)
        # print(self.filenames[idx])
        class_folder, frame_number = self.filenames[idx].lstrip().replace("\n", "").split(' ')

        if self.mode == 'train':
            rgb_path = os.path.join(self.args.data_path, class_folder, f"FrameBuffer_{frame_number}.png")
            depth_path = os.path.join(self.args.data_path, class_folder, f"Depth_{frame_number}.png")
        else:
            rgb_path = os.path.join(self.args.data_path_eval, class_folder, f"FrameBuffer_{frame_number}.png")
            depth_path = os.path.join(self.args.data_path_eval, class_folder, f"Depth_{frame_number}.png")

        # Load images
        image = Image.open(rgb_path).convert("RGB")
        depth_gt = Image.open(depth_path)

        # Crop if needed
        if self.args.do_kb_crop:
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

        # Preprocess images
        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)
        # print("Depth GT:", depth_gt.shape)
        # print("Image:", image.shape)
        normals = self.compute_normals_from_depth(depth_gt)

        if self.args.dataset == "nyu":
            depth_gt = depth_gt / 1000.0
        else:
            depth_gt = depth_gt / 256.0

        if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
            image, depth_gt, _, _ = self.random_crop(image, depth_gt, normals.copy(), self.args.input_height, self.args.input_width)

        # Compute normals dynamically
        # normals = self.compute_normals_from_depth(depth_gt.squeeze())

        # Preprocess for training
        image, depth_gt, normals = self.train_preprocess(image, depth_gt, normals)

        sample = {
            "image": image,
            "depth": depth_gt,
            "normal": normals,
            "focal": focal,
        }

        if self.transform:
            return self.transform((sample, self.args.dataset))
        return sample

    def __len__(self):
        return len(self.filenames)

    # @staticmethod
    # def compute_normals_from_depth(depth_array):
    #     """Compute surface normals from the depth map."""
    #
    #     rows, cols = depth_array.shape[:2]
    #
    #     x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    #     x = x.astype(np.float32)
    #     y = y.astype(np.float32)
    #
    #     # Calculate the partial derivatives of depth with respect to x and y
    #     dx = cv2.Sobel(depth_array, cv2.CV_32F, 1, 0)
    #     dy = cv2.Sobel(depth_array, cv2.CV_32F, 0, 1)
    #
    #     # Compute the normal vector for each pixel
    #     normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    #     norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    #     normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)
    #
    #     # Map the normal vectors to the [0, 255] range and convert to uint8
    #     normal = (normal + 1) * 127.5
    #     normal = normal.clip(0, 255).astype(np.uint8)
    #
    #     # Save the normal map to a file
    #     # normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    #
    #     dy, dx = np.gradient(depth_array)
    #     normals = np.dstack((-dx, -dy, np.ones_like(depth_array)))  # Negative gradients with Z=1
    #     norm = np.linalg.norm(normals, axis=2, keepdims=True)
    #     normals /= (norm + 1e-8)  # Normalize vectors
    #
    #     return normals

    @staticmethod
    def compute_normals_from_depth(depth_array):
        """Compute surface normals from the depth map."""

        depth_array = np.squeeze(depth_array)
        # print("Depth Array:", depth_array.shape)

        # Check if depth_array is valid
        if  depth_array.shape[0] < 2 or depth_array.shape[1] < 2:
            raise ValueError(f"Invalid depth_array shape: {depth_array.shape}. Must be at least 2x2.")

        # Calculate the partial derivatives of depth with respect to x and y
        dy, dx = np.gradient(depth_array)

        # Compute the normal vector for each pixel
        rows, cols = depth_array.shape[:2]
        normal = np.dstack((-dx, -dy, np.ones((rows, cols))))

        # Normalize the normal vectors
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

        return normal

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, normal, height, width):
        # print("Image:", img.shape)
        # print("Depth", depth.shape)
        # print("Normal", normal.shape)

        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        normal = normal[y:y + height, x:x + width, :]
        return img, depth, normal, x

    def train_preprocess(self, image, depth_gt, normal):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            normal = (normal[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt, normal

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def Cut_Flip(self, image, depth, normal):

        p = random.random()
        if p < 0.5:
            return image, depth, normal
        image_copy = copy.deepcopy(image)
        depth_copy = copy.deepcopy(depth)
        normal_copy = copy.deepcopy(normal)
        h, w, c = image.shape
        N = 2  # split numbers
        h_list = []
        h_interval_list = []  # hight interval
        for i in range(N - 1):
            h_list.append(random.randint(int(0.2 * h), int(0.8 * h)))
        h_list.append(h)
        h_list.append(0)
        h_list.sort()
        h_list_inv = np.array([h] * (N + 1)) - np.array(h_list)
        for i in range(len(h_list) - 1):
            h_interval_list.append(h_list[i + 1] - h_list[i])
        for i in range(N):
            image[h_list[i]:h_list[i + 1], :, :] = image_copy[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]
            depth[h_list[i]:h_list[i + 1], :, :] = depth_copy[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]
            normal[h_list[i]:h_list[i + 1], :, :] = normal_copy[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]

        return image, depth, normal


class ToTensor(object):
    def __init__(self, mode):
        print("ToTensor Initialized")
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # def __call__(self, sample_dataset):
    #     print("ToTensor called")
    #     print(sample_dataset)
    #     sample = sample_dataset[0]
    #     dataset = sample_dataset[1]
    #
    #     image, focal = sample['image'], sample['focal']
    #     H, W, _ = image.shape
    #     image = self.to_tensor(image)
    #     image = self.normalize(image)
    #
    #     if dataset == 'kitti':
    #         K = np.array([[716.88 / 4.0, 0, 596.5593 / 4.0, 0],
    #                       [0, 716.88 / 4.0, 149.854 / 4.0, 0],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1]], dtype=np.float32)
    #         K_p = np.array([[716.88, 0, 596.5593, 0],
    #                         [0, 716.88, 149.854, 0],
    #                         [0, 0, 1, 0],
    #                         [0, 0, 0, 1]], dtype=np.float32)
    #         inv_K = np.linalg.pinv(K)
    #         inv_K_p = np.linalg.pinv(K_p)
    #         K = torch.from_numpy(K)
    #         inv_K = torch.from_numpy(inv_K)
    #         inv_K_p = torch.from_numpy(inv_K_p)
    #
    #     elif dataset == 'nyu':
    #         K = np.array([[518.8579 / 4.0, 0, 325.5824 / 4.0, 0],
    #                       [0, 518.8579 / 4.0, 253.7362 / 4.0, 0],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1]], dtype=np.float32)
    #         K_p = np.array([[518.8579, 0, 325.5824, 0],
    #                         [0, 518.8579, 253.7362, 0],
    #                         [0, 0, 1, 0],
    #                         [0, 0, 0, 1]], dtype=np.float32)
    #         inv_K = np.linalg.pinv(K)
    #         inv_K_p = np.linalg.pinv(K_p)
    #         K = torch.from_numpy(K)
    #         inv_K = torch.from_numpy(inv_K)
    #         inv_K_p = torch.from_numpy(inv_K_p)
    #
    #     elif dataset == 'ucl':
    #         K = np.array([[518.8579 / 4.0, 0, 325.5824 / 4.0, 0],
    #                       [0, 518.8579 / 4.0, 253.7362 / 4.0, 0],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1]], dtype=np.float32)
    #         K_p = np.array([[518.8579, 0, 325.5824, 0],
    #                         [0, 518.8579, 253.7362, 0],
    #                         [0, 0, 1, 0],
    #                         [0, 0, 0, 1]], dtype=np.float32)
    #         inv_K = np.linalg.pinv(K)
    #         inv_K_p = np.linalg.pinv(K_p)
    #         K = torch.from_numpy(K)
    #         inv_K = torch.from_numpy(inv_K)
    #         inv_K_p = torch.from_numpy(inv_K_p)
    #
    #     if self.mode == 'test':
    #         return {'image': image, 'K': K, 'inv_K': inv_K, 'inv_K_p': inv_K_p, 'focal': focal}
    #
    #     depth = sample['depth']
    #     if self.mode == 'train':
    #         depth = self.to_tensor(depth)
    #         normal = sample['normal']
    #         normal = self.to_tensor(normal)
    #         if dataset == 'kitti':
    #             K = np.array([[716.88 / 4.0, 0, 596.5593 / 4.0, 0],
    #                           [0, 716.88 / 4.0, 149.854 / 4.0, 0],
    #                           [0, 0, 1, 0],
    #                           [0, 0, 0, 1]], dtype=np.float32)
    #             K[0][2] -= sample['offset'] / 4.0
    #             K_p = np.array([[716.88, 0, 596.5593, 0],
    #                             [0, 716.88, 149.854, 0],
    #                             [0, 0, 1, 0],
    #                             [0, 0, 0, 1]], dtype=np.float32)
    #             K_p[0][2] -= sample['offset']
    #             inv_K = np.linalg.pinv(K)
    #             inv_K_p = np.linalg.pinv(K_p)
    #
    #             K = torch.from_numpy(K)
    #             K_p = torch.from_numpy(K_p)
    #             inv_K = torch.from_numpy(inv_K)
    #             inv_K_p = torch.from_numpy(inv_K_p)
    #         elif dataset == 'nyu':
    #             K = np.array([[518.8579 / 4.0, 0, 325.5824 / 4.0, 0],
    #                           [0, 518.8579 / 4.0, 253.7362 / 4.0, 0],
    #                           [0, 0, 1, 0],
    #                           [0, 0, 0, 1]], dtype=np.float32)
    #             K_p = np.array([[518.8579, 0, 325.5824, 0],
    #                             [0, 518.8579, 253.7362, 0],
    #                             [0, 0, 1, 0],
    #                             [0, 0, 0, 1]], dtype=np.float32)
    #             inv_K = np.linalg.pinv(K)
    #             inv_K_p = np.linalg.pinv(K_p)
    #
    #             K = torch.from_numpy(K)
    #             K_p = torch.from_numpy(K_p)
    #             inv_K = torch.from_numpy(inv_K)
    #             inv_K_p = torch.from_numpy(inv_K_p)
    #         elif dataset == 'ucl':
    #             print("UCL Dataset")
    #             K = np.array([[518.8579 / 4.0, 0, 325.5824 / 4.0, 0],
    #                           [0, 518.8579 / 4.0, 253.7362 / 4.0, 0],
    #                           [0, 0, 1, 0],
    #                           [0, 0, 0, 1]], dtype=np.float32)
    #             K_p = np.array([[518.8579, 0, 325.5824, 0],
    #                             [0, 518.8579, 253.7362, 0],
    #                             [0, 0, 1, 0],
    #                             [0, 0, 0, 1]], dtype=np.float32)
    #             inv_K = np.linalg.pinv(K)
    #             inv_K_p = np.linalg.pinv(K_p)
    #             K = torch.from_numpy(K)
    #             inv_K = torch.from_numpy(inv_K)
    #             inv_K_p = torch.from_numpy(inv_K_p)
    #         return {'image': image, 'depth': depth, 'normal': normal, 'K': K, 'inv_K': inv_K, 'inv_K_p': inv_K_p,
    #                 'focal': focal}
    #     else:
    #         has_valid_depth = sample['has_valid_depth']
    #         return {'image': image, 'depth': depth, 'K': K, 'inv_K': inv_K, 'inv_K_p': inv_K_p, 'focal': focal,
    #                 'has_valid_depth': has_valid_depth}

    def __call__(self, sample_dataset):
        sample, dataset = sample_dataset  # Unpack the sample and dataset type

        # Extract common fields
        image, focal = sample['image'], sample['focal']
        H, W, _ = image.shape
        image = self.to_tensor(image)
        image = self.normalize(image)

        # Initialize K and inv_K based on the dataset
        if dataset == 'kitti':
            K = np.array([[716.88 / 4.0, 0, 596.5593 / 4.0, 0],
                          [0, 716.88 / 4.0, 149.854 / 4.0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
            K_p = np.array([[716.88, 0, 596.5593, 0],
                            [0, 716.88, 149.854, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        elif dataset == 'nyu' or dataset == 'ucl':
            K = np.array([[518.8579 / 4.0, 0, 325.5824 / 4.0, 0],
                          [0, 518.8579 / 4.0, 253.7362 / 4.0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
            K_p = np.array([[518.8579, 0, 325.5824, 0],
                            [0, 518.8579, 253.7362, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Compute inverse K matrices
        inv_K = np.linalg.pinv(K)
        inv_K_p = np.linalg.pinv(K_p)
        K = torch.from_numpy(K)
        K_p = torch.from_numpy(K_p)
        inv_K = torch.from_numpy(inv_K)
        inv_K_p = torch.from_numpy(inv_K_p)
        if self.mode == 'test':
            return {'image': image, 'K': K, 'inv_K': inv_K, 'inv_K_p': inv_K_p, 'focal': focal}

        # Handle depth and normal for training
        depth = self.to_tensor(sample['depth'])
        normal = self.to_tensor(sample['normal'])

        if self.mode == 'train':
            # Specific offset adjustment for KITTI if needed
            if dataset == 'kitti' and 'offset' in sample:
                K[0, 2] -= sample['offset'] / 4.0
                K_p[0, 2] -= sample['offset']
                inv_K = torch.from_numpy(np.linalg.pinv(K.numpy()))
                inv_K_p = torch.from_numpy(np.linalg.pinv(K_p.numpy()))

            return {
                'image': image,
                'depth': depth,
                'normal': normal,
                'K': K,
                'inv_K': inv_K,
                'inv_K_p': inv_K_p,
                'focal': focal
            }
        else:
            # has_valid_depth = sample['has_valid_depth']
            has_valid_depth = True
            return {
                'image': image,
                'depth': depth,
                'K': K,
                'inv_K': inv_K,
                'inv_K_p': inv_K_p,
                'focal': focal,
                'has_valid_depth': has_valid_depth
            }

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img