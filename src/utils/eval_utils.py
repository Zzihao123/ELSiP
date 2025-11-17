import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc, MIL_Mean, MIL_Max, MIL_Attention
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_our import *
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'our_clam_sb':
        if not hasattr(args, 'task') or args.task is None:
            model = ELSiP(**model_dict)
        else:
            model = ELSiP(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        if 'patch_encoder' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})

    try:
        model.load_state_dict(ckpt_clean, strict=True)
        print("Successfully loaded checkpoint with strict=True")
    except RuntimeError as e:
        print("Attempting to load with strict=False...")
        model_state_dict = model.state_dict()
        filtered_ckpt = {}
        for key, value in ckpt_clean.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                filtered_ckpt[key] = value
        missing_keys, unexpected_keys = model.load_state_dict(filtered_ckpt, strict=False)
        print(f"Loaded with strict=False. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        if missing_keys:
            print("Missing keys:", missing_keys[:5], "..." if len(missing_keys) > 5 else "")
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys[:5], "..." if len(unexpected_keys) > 5 else "")

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    if args.model_type == 'our_clam_sb':
        patient_results, test_error, auc, df, _ = summary_our(model, loader, args, enhance=True)
    else:
        patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger

def summary_our(model, loader, args, enhance=False):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, batch_data in enumerate(loader):
        # Handle different data formats from collate function
        slide_features, patch_features, label, clinical_text, slide_coords, patch_coords = batch_data
        # Move tensors to device (clinical_text is a list of strings, not a tensor)
        slide_features = slide_features.to(device)
        patch_features = patch_features.to(device)
        label = label.to(device)
        patch_coords = torch.from_numpy(patch_coords).to(device)
        # clinical_text stays as list, slide_coords and patch_coords are numpy arrays


        # Pass data to model
        # logits, Y_prob, Y_hat, _, instance_dict = model(slide_features, patch_features, clinical_text)
        import h5py
        my_h5_path = "/data/zzh/WSI_zzh/trident/trident_processed_grandqc/20x_512px_0px_overlap/features_conch_v15/0_1.h5"
        file = h5py.File(my_h5_path, 'r')

        patch_size_lv0 = file['coords'].attrs['patch_size_level0']

        # clinical_text stays as list, slide_coords and patch_coords are numpy arrays

        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():

            if enhance:
                logits, Y_prob, Y_hat, _, instance_dict = model(
                    slide_features, 
                    patch_features, 
                    clinical_text, 
                    patch_coords=patch_coords,  
                    patch_size_lv0=patch_size_lv0, 
                    enhance=True
                )
            else:
                logits, Y_prob, Y_hat, _, _ = model(slide_features, patch_features, clinical_text)

            # logits, Y_prob, Y_hat, _, _ = model(slide_features, patch_features, clinical_text)


        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del batch_data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger