import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")
np.set_printoptions(suppress=True)

from aidia import image
from aidia import utils
from aidia.ai.dataset import Dataset
from aidia.ai.config import AIConfig
from aidia.ai.models.unet import UNet
from aidia.ai import metrics


class SegmentationModel(object):
    def __init__(self, config:AIConfig) -> None:
        self.config = config
        self.dataset = None
        self.model = None

    def set_config(self, config):
        self.config = config

    def build_dataset(self):
        self.dataset = Dataset(self.config)
    
    def load_dataset(self):
        self.dataset = Dataset(self.config, load=True)
    
    def build_model(self, mode, weights_path=None):
        assert mode in ["train", "test"]

        # Build UNet model
        model = UNet(self.config.num_classes)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)
        
        if mode == 'train':
            # Setup optimizer and loss
            self.optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
            self.criterion = nn.BCELoss()
            self.model = model
            return model
        
        if mode == "test":
            if weights_path is None or not os.path.exists(weights_path):
                raise ValueError("weights_path must be provided for test mode.")
            checkpoint = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval()
            self.model = model
            return model
           
    def train(self, custom_callbacks=None):
        """Train the model with the dataset."""
        checkpoint_dir = utils.get_dirpath_with_mkdir(self.config.log_dir, "weights")

        train_dataloader, val_dataloader = self.get_pytorch_dataloaders()

        if custom_callbacks is not None:
            on_train_batch_end = custom_callbacks[0]
            on_val_end = custom_callbacks[1]

        best_val_loss = float('inf')
        
        for epoch in range(self.config.EPOCHS):
            # Training phase
            self.model.train()
            train_loss = 0.0
            batch_num = 0
            tmp_imgs = []
            
            for batch_idx, (images, masks) in enumerate(train_dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if epoch == 0 and batch_idx < 10:
                    tmp_imgs.append(images.cpu().numpy()[0].transpose(1,2,0)*255)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                batch_num += 1
            
                avg_train_loss = train_loss / batch_num
                on_train_batch_end(avg_train_loss)

            if epoch == 0:
                tmp_dir = utils.get_dirpath_with_mkdir(self.config.log_dir, "train_images")
                for i, _img in enumerate(tmp_imgs):
                    image.imwrite(
                        np.array(_img, dtype=np.uint8),
                        os.path.join(tmp_dir, f"train_input_{i}.png"))
            
            # Validation phase
            self.model.eval()
            val_batch_num = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for images, masks in val_dataloader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    val_loss += loss.item()
                    val_batch_num += 1
            
            avg_val_loss = val_loss / val_batch_num
            on_val_end(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(self.model.state_dict(), best_path)
        
        # Save last model
        last_path = os.path.join(checkpoint_dir, "last.pt")
        torch.save(self.model.state_dict(), last_path)
    
    
    def get_pytorch_dataloaders(self):
        """Get PyTorch DataLoaders for training and validation."""
        train_dataset = SegDataset(self.dataset, self.config, mode="train")
        val_dataset = SegDataset(self.dataset, self.config, mode="val")

        g = torch.Generator()
        g.manual_seed(self.config.SEED)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.total_batchsize,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            generator=g,
            persistent_workers=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.total_batchsize,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=g,
            persistent_workers=False,
        )
        
        return train_loader, val_loader

    # def stop_training(self):
    #     self.model.stop_training = True

    def evaluate(self):
        res = {}
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        eval_dict = {}
        eval_dict["Metrics"] = [
            "Accuracy", "Precision", "Recall", "Specificity",
            "F1", "ROC Curve AUC", "PR Curve AUC (Average Precision)",
        ]

        # prepare labels
        num_classes = self.config.num_classes + 1
        labels = self.config.LABELS[:]
        labels.insert(0, 'background')  # add background class

        eval_dir = utils.get_dirpath_with_mkdir(self.config.log_dir, 'evaluation')
        roc_pr_dir = utils.get_dirpath_with_mkdir(eval_dir, 'roc_pr_fig')

        # THRESHOLDS = [0.00001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99999]
        THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        tptnfpfn_per_class = np.zeros((num_classes, len(THRESHOLDS),  4), int)
        eval_per_class = np.zeros((num_classes, len(eval_dict["Metrics"])), float)
        cm_multi_class = np.zeros((num_classes, num_classes), float)

        # predict all test data
        for i, image_id in enumerate(self.dataset.test_ids):
            # for DEBUG
            # if i > 50:
            #     break

            # predict
            img = self.dataset.load_image(image_id)
            mask = self.dataset.load_masks(image_id)
            inputs = image.preprocessing(img, is_tensor=True, channel_first=True, is_norm=True)
            
            # Convert to tensor and move to device
            inputs_tensor = torch.from_numpy(inputs).float().to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                pred = self.model(inputs_tensor)
            pred = pred.cpu().numpy()[0]
            
            y_true = np.array(mask)
            y_pred = np.array(pred)

            # multiclass confusion matrix
            yt_label = y_true.argmax(axis=2).ravel()
            yp_label = y_pred.argmax(axis=0).ravel()
            cm_multi_class += metrics.multi_confusion_matrix(yt_label, yp_label, num_classes)

            # calculate tp tn fp fn per class
            for class_id in range(num_classes):
                # get result by class
                yt_class = y_true[..., class_id]
                yp_class = y_pred[class_id]

                for i, th in enumerate(THRESHOLDS):
                    # thresholding
                    _yp_class = np.copy(yp_class)
                    _yp_class[_yp_class >= th] = 1
                    _yp_class[_yp_class < th] = 0
                    _yp_class = _yp_class.astype(np.uint8)

                    if np.max(yt_class) == 0 and np.max(_yp_class) == 0: # no ground truth
                        tn, fp, fn, tp = self.config.INPUT_SIZE**2, 0, 0, 0
                    else:
                        cm = metrics.binary_confusion_matrix(yt_class.ravel(), _yp_class.ravel())
                        tn, fp, fn, tp = cm.ravel()
                    _cm = np.array([tp, tn, fp, fn])
                    tptnfpfn_per_class[class_id, i] += _cm

        # calculate evaluation values per class
        for class_id in range(len(labels)):
            class_name = labels[class_id]
            fpr = [0.0]
            tpr = [0.0]
            pres = [1.0]
            recs = [0.0]
            auc = 0
            ap = 0
            result_at_05 = None
            for i, thresh in enumerate(THRESHOLDS):
                tp, tn, fp, fn = tptnfpfn_per_class[class_id, i]
                accuracy, precision, recall, specificity, fscore = metrics.common_metrics(tp, tn, fp, fn)

                 # save result at threshold=0.5
                if thresh == 0.5:
                    result_at_05 = [accuracy, precision, recall, specificity, fscore]

                # if precision == 0 and recall == 0:
                #     precision = 1

                tpr.append(recall)
                fpr.append(fpr[-1])
                tpr.append(recall)
                fpr.append(1 - specificity)

                pres.append(pres[-1])
                recs.append(recall)
                pres.append(precision)
                recs.append(recall)

                auc += abs(fpr[-1] - fpr[-3]) * tpr[-1]
                ap += abs(recs[-1] - recs[-3]) * pres[-3]
                  
            # add the last value
            tpr.append(1.0)
            fpr.append(fpr[-1])
            tpr.append(1.0)
            fpr.append(1.0)
            auc += abs(fpr[-1] - fpr[-3]) * tpr[-1]

            pres.append(pres[-1])
            recs.append(1.0)
            pres.append(0.0)
            recs.append(1.0)
            ap += abs(recs[-1] - recs[-3]) * pres[-3]

            # draw curves
            ax.plot([0.0, 1.0], [0.0, 1.0], color='k', linestyle='--', label='baseline')
            ax.plot(fpr, tpr, marker='o', color='red', label='ours')
            ax.text(0.8, 0.2, f'AUC = {auc:.02f}', ha='center', color='red', fontsize=16)
            ax.set_title(f"ROC Curve ({class_name})", fontsize=20)
            ax.set_xlabel('FPR', fontsize=16)
            ax.set_ylabel('TPR', fontsize=16)
            ax.legend(fontsize=16)
            ax.grid()
            fig.savefig(os.path.join(roc_pr_dir, f"{class_name}_roc.png"))
            ax.clear()

            ax.plot([0.0, 1.0], [1.0, 0.0], color='k', linestyle='--', label='baseline')
            ax.plot(recs, pres, marker='o', color='red', label='ours')
            ax.set_title(f"PR Curve ({class_name})", fontsize=20)
            ax.set_xlabel('Recall', fontsize=16)
            ax.set_ylabel('Precision', fontsize=16)
            ax.text(0.2, 0.2, f'AUC = {ap:.02f}', ha='center', color='red', fontsize=16)
            ax.legend(fontsize=16)
            ax.grid()
            fig.savefig(os.path.join(roc_pr_dir, f"{class_name}_pr.png"))
            ax.clear()

            # add result per class
            eval_dict[class_name] = result_at_05 + [auc, ap]
            eval_per_class[class_id] = result_at_05 + [auc, ap]
        
        # macro mean
        acc = np.mean(eval_per_class[..., 0])
        pre = np.mean(eval_per_class[..., 1])
        rec = np.mean(eval_per_class[..., 2])
        spe = np.mean(eval_per_class[..., 3])
        fscore = np.mean(eval_per_class[..., 4])
        auc = np.mean(eval_per_class[..., 5])
        ap = np.mean(eval_per_class[..., 6])

        # confusion matrix
        cm_multi_class = cm_multi_class / (np.sum(cm_multi_class, axis=1) + 1e-12)[:, None]

        # figure of confusion matrix
        ax.set_title('Confusion Matrix', fontsize=20)
        metrics.confusion_matrix_display(fig, ax, 
                                         confusion_matrix=cm_multi_class,
                                         display_labels=labels)
        filename = os.path.join(eval_dir, "confusion_matrix.png")
        fig.savefig(filename)
        img = image.fig2img(fig)

        # save eval dict
        eval_dict["(Macro Mean)"] = [acc, pre, rec, spe, fscore, auc, ap]
        utils.save_dict_to_excel(eval_dict, os.path.join(eval_dir, "scores.xlsx"))

        res = {
            "Accuracy": acc,
            "Precision": pre,
            "Recall": rec,
            "Specificity": spe,
            "F-Score": fscore,
            "ROC Curve AUC": auc,
            "PR Curve AUC (Average Precision)": ap,
            "img": img,
        }
        return res

    def predict_by_id(self, image_id, thresh=0.5):
        src_img = self.dataset.load_image(image_id)
        gt_mask_data = self.dataset.load_masks(image_id)
        img = image.preprocessing(src_img, is_tensor=True, channel_first=True, is_norm=True)
        
        # Convert to tensor and move to device
        img_tensor = torch.from_numpy(img).float().to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            pred = self.model(img_tensor)
        pred = pred.cpu().numpy()[0]
        
        concat = image.mask2merge(src_img, pred, self.dataset.class_names, gt_mask_data, thresh)
        return concat
    
    def convert2onnx(self):
        """Convert the PyTorch model to ONNX format."""
        onnx_path = os.path.join(self.config.log_dir, "model.onnx")
        try:
            self.model.eval()
            dummy_input = torch.randn(1, 3, self.config.INPUT_SIZE, 
                                     self.config.INPUT_SIZE).to(self.device)
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        except Exception as e:
            print(f"Failed to convert model to ONNX: {e}")
            return False
        return True

    def set_inference_model(self, weights_path=None):
        """Set the model for inference mode."""
        if weights_path is None:
            weights_path = os.path.join(self.config.log_dir, "weights", "best.pt")
        self.model = self.build_model("test", weights_path=weights_path)
        return self.model




class SegDataset(TorchDataset):
    """PyTorch Dataset for segmentation tasks."""
    
    def __init__(self, dataset: Dataset, config: AIConfig, mode="train") -> None:
        assert mode in ["train", "val", "test"]
        
        self.dataset = dataset
        self.config = config
        self.mode = mode
        
        # Select appropriate image IDs based on mode
        if mode == "train":
            self.image_ids = self.dataset.train_ids
        elif mode == "val":
            self.image_ids = self.dataset.val_ids
        else:  # test
            self.image_ids = self.dataset.test_ids
            
        if mode == "train":
            # Shuffle training data
            np.random.shuffle(self.image_ids)

            # augmentations
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5) if self.config.RANDOM_HFLIP else T.Identity(),
                T.RandomVerticalFlip(p=0.5) if self.config.RANDOM_VFLIP else T.Identity(),
                T.RandomAffine(
                    degrees=self.config.RANDOM_ROTATE * 180 if self.config.RANDOM_ROTATE > 0.0 else 0.0,
                    translate=(self.config.RANDOM_SHIFT, self.config.RANDOM_SHIFT) if self.config.RANDOM_SHIFT > 0.0 else (0.0, 0.0),
                    scale=(1 - self.config.RANDOM_SCALE, 1 + self.config.RANDOM_SCALE) if self.config.RANDOM_SCALE > 0.0 else (1.0, 1.0),
                    shear=self.config.RANDOM_SHEAR * 40 if self.config.RANDOM_SHEAR > 0.0 else 0.0,
                    fill=0.5,
                    ),
                T.ColorJitter(
                    brightness=self.config.RANDOM_BRIGHTNESS if self.config.RANDOM_BRIGHTNESS > 0.0 else 0.0,
                    contrast=self.config.RANDOM_CONTRAST if self.config.RANDOM_CONTRAST > 0.0 else 0.0,
                    saturation=0,
                    hue=0,
                    ),
                T.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, self.config.RANDOM_BLUR * 20.0),) if self.config.RANDOM_BLUR > 0.0 else T.Identity(),
                T.GaussianNoise(
                    mean=0.0,
                    sigma=random.uniform(0.0, self.config.RANDOM_NOISE * 0.1),) if self.config.RANDOM_NOISE > 0.0 else T.Identity(),
            ])
        
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image and mask
        img = self.dataset.load_image(image_id)
        masks = self.dataset.load_masks(image_id)

        # Convert to torch tensors
        img_tensor = torch.from_numpy(img.astype(np.float32))
        mask_tensor = torch.from_numpy(masks.astype(np.float32))

        # Ensure correct dimensions (H, W, C) -> (C, H, W)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.permute(2, 0, 1)

        img_tensor /= 255.0  # Normalize to [0, 1]
        
        if self.mode == "train":
            mask_tensor = tv_tensors.Mask(mask_tensor)
            img_tensor, mask_tensor = self.transforms(img_tensor, mask_tensor)
            if img_tensor is None or mask_tensor is None:
                raise ValueError("Transformations resulted in None tensor.")
            
        return img_tensor, mask_tensor
    
    def on_epoch_end(self):
        """Called at the end of each epoch for training mode."""
        if self.mode == "train":
            np.random.shuffle(self.image_ids)


