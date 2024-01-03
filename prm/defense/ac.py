import torch
from tqdm import tqdm

def get_features(data_loader, model, num_classes):

    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    feats = []

    with torch.no_grad():
        sid = 0
        for (ins_data, ins_target) in tqdm(data_loader):
            ins_data = ins_data.cuda()
            _, x_feats = model(ins_data, True)
            batch_size = len(ins_target)
            for bid in range(batch_size):
                feats.append(x_feats[bid])
                b_target = ins_target[bid].item()
                class_indices[b_target].append(sid + bid)

            sid += batch_size

    return feats, class_indices

def cleanser(inspection_set, model, num_classes, args, clusters=2):

    from sklearn.cluster import KMeans

    inspection_split_loader = torch.utils.data.DataLoader(inspection_set, batch_size=128, shuffle=False)

    suspicious_indices = []
    feats, class_indices = get_features(inspection_split_loader, model, num_classes)
    
    for target_class in range(num_classes):

        if len(class_indices[target_class]) <= 1:
            continue
        
        temp_feats = [feats[temp_idx].unsqueeze(dim=0) for temp_idx in class_indices[target_class]]
        temp_feats = torch.cat( temp_feats , dim=0)
        temp_feats = temp_feats - temp_feats.mean(dim=0)

        _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

        axes = V[:, :10]
        projected_feats = torch.matmul(temp_feats, axes)
        projected_feats = projected_feats.cpu().numpy()
        kmeans = KMeans(n_clusters=2, n_init=10).fit(projected_feats)

        if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
            clean_label = 1
        else:
            clean_label = 0

        outliers = []
        for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
            if bool:
                outliers.append(class_indices[target_class][idx])

        if len(outliers) < len(kmeans.labels_) * 0.35:
            print(f"Outlier Num in Class {target_class}:", len(outliers))
            suspicious_indices += outliers

        return suspicious_indices
        