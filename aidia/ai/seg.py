import os
import tensorflow as tf
import numpy as np
import glob
import random
import imgaug
import tf2onnx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")
plt.rcParams["font.size"] = 15
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

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        tf.random.set_seed(self.config.SEED)
        imgaug.seed(self.config.SEED)

    def set_config(self, config):
        self.config = config

    def build_dataset(self):
        self.dataset = Dataset(self.config)
    
    def load_dataset(self):
        self.dataset = Dataset(self.config, load=True)
    
    def build_model(self, mode, weights_path=None):
        assert mode in ["train", "test"]
        self.model = UNet(self.config.num_classes)

        input_shape = (None, self.config.INPUT_SIZE, self.config.INPUT_SIZE, 3)
        self.model.build(input_shape=input_shape)
        self.model.compute_output_shape(input_shape=input_shape)

        optim = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        self.model.compile(optimizer=optim, loss=tf.keras.losses.BinaryCrossentropy())
        
        if mode == "test":
            if weights_path and os.path.exists(weights_path):
                self.model.load_weights(weights_path)
            else:
                _wlist = os.path.join(self.config.log_dir, "weights", "*.h5")
                weights_path = sorted(glob.glob(_wlist))[-1]
                self.model.load_weights(weights_path)


    def train(self, custom_callbacks=None):
        checkpoint_dir = utils.get_dirpath_with_mkdir(
            self.config.log_dir, "weights"
        )
        if self.config.SAVE_BEST:
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "{epoch:04d}.h5")

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=self.config.SAVE_BEST,
                save_weights_only=True,
                period=1 if self.config.SAVE_BEST else 20, # TODO:user setting
            ),
        ]
        if custom_callbacks:
            for c in custom_callbacks:
                callbacks.append(c)

        train_generator = SegDataGenerator(self.dataset, self.config, mode="train")
        val_generator = SegDataGenerator(self.dataset, self.config, mode="val")

        train_generator = tf.data.Dataset.from_generator(
            train_generator.flow, (tf.float32, tf.float32),
            output_shapes=(self.model.input_shape, self.model.output_shape)
        )
        val_generator = tf.data.Dataset.from_generator(
            val_generator.flow, (tf.float32, tf.float32),
            output_shapes=(self.model.input_shape, self.model.output_shape)
        )
        
        self.model.fit(
            train_generator,
            steps_per_epoch=self.dataset.train_steps,
            epochs=self.config.EPOCHS,
            verbose=0,
            validation_data=val_generator,
            validation_steps=self.dataset.val_steps,
            callbacks=callbacks
        )

        # save last model
        if not self.config.SAVE_BEST:
            checkpoint_path = os.path.join(checkpoint_dir, "last_model.h5")
            self.model.save_weights(checkpoint_path)


    def stop_training(self):
        self.model.stop_training = True

    def evaluate(self, cb_widget):
        res = {}
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        eval_dict = {}
        eval_dict["Metrics"] = [
            "Accuracy", "Precision", "Recall", "Specificity",
            "F1", "ROC Curve AUC", "Average Precision",
        ]

        # prepare labels
        num_classes = self.config.num_classes + 1
        labels = self.config.LABELS[:]
        labels.insert(0, 'no label')

        eval_dir = utils.get_dirpath_with_mkdir(
            self.config.log_dir, 'evaluation'
        )
        roc_pr_dir = utils.get_dirpath_with_mkdir(
            eval_dir, 'roc_pr_fig'
        )

        tptnfpfn_per_class = np.zeros((num_classes, 9,  4), int)
        eval_per_class = np.zeros((num_classes, len(eval_dict["Metrics"])), float)
        cm_multi_class = np.zeros((num_classes, num_classes), float)

        # predict all test data
        for i, image_id in enumerate(self.dataset.test_ids):
            cb_widget.notifyMessage.emit(f"Evaluating... {i+1} / {self.dataset.num_test}")
            cb_widget.progressValue.emit(int((i+1) / self.dataset.num_test * 99))

            # predict
            img = self.dataset.load_image(image_id)
            mask = self.dataset.load_masks(image_id)
            inputs = image.preprocessing(img, is_tensor=True)
            pred = self.model.predict_on_batch(inputs)[0]
            y_true = np.array(mask)
            y_pred = np.array(pred)

            # multiclass confusion matrix
            yt_label = y_true.argmax(axis=2).ravel()
            yp_label = y_pred.argmax(axis=2).ravel()
            cm_multi_class += metrics.multi_confusion_matrix(yt_label, yp_label, num_classes)

            # calculate tp tn fp fn per class
            for i, th in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
                for class_id in range(num_classes):
                    # get result by class
                    yt_class = y_true[..., class_id]
                    yp_class = y_pred[..., class_id]

                    # thresholding
                    yp_class[yp_class >= th] = 1
                    yp_class[yp_class < th] = 0
                    yp_class = yp_class.astype(np.uint8)

                    if np.max(yt_class) == 0 and np.max(yp_class) == 0: # no ground truth
                        tn, fp, fn, tp = self.config.INPUT_SIZE**2, 0, 0, 0
                    else:
                        cm = confusion_matrix(yt_class.ravel(), yp_class.ravel())
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
            for i in range(9):
                tp, tn, fp, fn = tptnfpfn_per_class[class_id, i]
                accuracy, precision, recall, specificity, fscore = metrics.common_metrics(tp, tn, fp, fn)
                fpr.append(1 - specificity)
                tpr.append(recall)
                pres.append(precision)
                recs.append(recall)
                auc += abs(fpr[i+1] - fpr[i]) * tpr[i+1]
                ap += abs(recs[i+1] - recs[i]) * pres[i]

                # save result at threshold=0.5
                if i == 4:
                    result_at_05 = [accuracy, precision, recall, specificity, fscore]
                  
            # add the last value
            fpr.append(1.0)
            tpr.append(1.0)
            auc += abs(fpr[-1] - fpr[-2]) * tpr[-1]

            pres.append(0.0)
            recs.append(1.0)
            ap += abs(recs[-1] - recs[-2]) * pres[-2]

            # draw curves
            ax.plot(fpr, tpr)
            ax.set_title(f"ROC Curve ({class_name})")
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.grid()
            fig.savefig(os.path.join(roc_pr_dir, f"{class_name}_roc.png"))
            ax.clear()

            ax.plot(recs, pres)
            ax.set_title(f"PR Curve ({class_name})")
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
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
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm_multi_class,
                                         display_labels=labels)
        cm_disp.plot(ax=ax)
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
            "Average Precision": ap,
            "img": img,
        }

        cb_widget.notifyMessage.emit("Done")
        cb_widget.progressValue.emit(100)

        return res

    def predict_by_id(self, image_id, thresh=0.5):
        src_img = self.dataset.load_image(image_id)
        gt_mask_data = self.dataset.load_masks(image_id)
        img = image.preprocessing(src_img, is_tensor=True)
        pred = self.model.predict(img, batch_size=1, verbose=0)[0]
        concat = image.mask2merge(src_img, pred, self.dataset.class_names, gt_mask_data, thresh)
        return concat
    
    def convert2onnx(self):
        onnx_path = os.path.join(self.config.log_dir, "model.onnx")
        if os.path.exists(onnx_path):
            return
        tf2onnx.convert.from_keras(self.model, opset=11, output_path=onnx_path)


class SegDataGenerator(object):
    def __init__(self, dataset:Dataset, config:AIConfig, mode="train") -> None:
        assert mode in ["train", "val", "test"]

        self.dataset = dataset
        self.config = config
        self.mode = mode

        self.augseq = config.get_augseq()
        self.images = []
        self.targets = []

        self.image_ids = self.dataset.train_ids
        self.augmentation = True
        if self.mode == "val":
            self.image_ids = self.dataset.val_ids
            self.augmentation = False
        if self.mode == "test":
            self.image_ids = self.dataset.test_ids
            self.augmentation = False
        np.random.shuffle(self.image_ids)

    def reset(self):
        self.images.clear()
        self.targets.clear()

    def flow(self):
        b = 0
        i = 0

        while True:
            image_id = self.image_ids[i]
            i += 1
            if i >= len(self.image_ids):
                i = 0
                np.random.shuffle(self.image_ids)

            img = self.dataset.load_image(image_id)
            masks = self.dataset.load_masks(image_id)

            if self.augmentation:
                img, masks = self.augment_image(img, masks)
                # if self.config.RANDOM_BRIGHTNESS > 0:
                    # img = self.random_brightness(img)
            
            self.images.append(img)
            self.targets.append(masks)

            b += 1

            if b >= self.config.total_batchsize:
                inputs = np.asarray(self.images, dtype=np.float32)
                inputs = inputs / 255.0
                outputs = np.asarray(self.targets, dtype=np.float32)
                yield inputs, outputs
                b = 0
                self.reset()


    def _hook(self, images, augmenter, parents, default):
        """Determines which augmenters to apply to masks."""
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                       "Fliplr", "Flipud", "CropAndPad",
                       "Affine", "PiecewiseAffine"]
        return augmenter.__class__.__name__ in MASK_AUGMENTERS
    

    def augment_image(self, img, masks):
        det = self.augseq.to_deterministic()
        img = det.augment_image(img)
        # only apply mask augmenters to masks
        res = []
        for class_id in range(masks.shape[2]):
            m = masks[:, :, class_id]
            m = det.augment_image(m, hooks=imgaug.HooksImages(activator=self._hook))
            res.append(m)
        res = np.stack(res, axis=2)
        return img, res

