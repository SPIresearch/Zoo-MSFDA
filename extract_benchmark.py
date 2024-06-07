<<<<<<< HEAD
import os

def split_dset_structure_model_source_target(file_name,net_set):
    file_name=file_name.split('.mat')[0]
    target=file_name[-1]
    source=file_name[-3]
    file_name=file_name[:-4]
    dset=file_name.split('_')[0]
    r=len(dset)
    model_name=file_name[r:]

    model_structure=model_name.split('_')[0]
    return dset,model_structure,model_name,source,target
def extract_targetdomain_features(folder_path, dataset_name,config,target_domain,net_set,save_path):
    
    files = sorted(os.listdir(folder_path))

    
  
    target_files = [file for file in files if f"_{target_domain}.mat" in file and file.split('_')[0]==dataset_name and file[-7]!=target_domain]  
   


    if not os.path.exists( f"{save_path}/{dataset_name}_{config}"):
        os.makedirs( f"{save_path}/{dataset_name}_{config}")

 
    output_filename = f"{save_path}/{dataset_name}_{config}/{target_domain}.txt"
    with open(output_filename, 'w') as output_file:
        for file_name in target_files:
            dset,model_structure,model_name,source,target=split_dset_structure_model_source_target(file_name,net_set)
            if model_name not in net_set:
                continue
            output_file.write(file_name + '\n')

    print(f"saved in {output_filename}")


if __name__ == "__main__":
    save_path = "./configs/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    folder_path = "" #the feature_save_dir in train_source.py
    dataset_name='office-home'

    net_set=['_resnet50','_resnet101','_efficientnet_v2_l','_efficientnet_v2_m','_efficientnet_v2_s','_swin_b','_swin_l',
             '_swin_s','_swin_t','_vit_b_16','_vit_b_32','_vit_l_16','_vit_l_32','_vit_h_14','_swin_v2_t','_swin_v2_s','_swin_v2_b']
    config='main'
    for target_domain in range(4):
        target_domain=str(target_domain)
        extract_targetdomain_features(folder_path,dataset_name, config,target_domain,net_set,save_path)


    # config='s1'
    # net_set=['_resnet50_1e-05','_resnet50_0.0001','_resnet50_0.001','_resnet50_0.003','_resnet50_0.01','_resnet50_0.03','_resnet50_0.1']
    # for target_domain in range(4):
    #     target_domain=str(target_domain)
    #     extract_targetdomain_features(folder_path,dataset_name, config,target_domain,net_set)

    # config='s2'
    # net_set=['_resnet50_16','_resnet50_32','_resnet50_128','_resnet50_64','_resnet50_256']
    # for target_domain in range(4):
    #     target_domain=str(target_domain)
    #     extract_targetdomain_features(folder_path,dataset_name, config,target_domain,net_set)

    # config='s3'
    # net_set=['_resnet50_Adam','_resnet50_AdamW','_resnet50_ASGD','_resnet50']
    # for target_domain in range(4):
    #     target_domain=str(target_domain)
    #     extract_targetdomain_features(folder_path,dataset_name, config,target_domain,net_set)


    # config='s4'
    # net_set=['_resnet50_v1','_resnet50_nopre','_resnet50']
    # for target_domain in range(4):
    #     target_domain=str(target_domain)
    #     extract_targetdomain_features(folder_path,dataset_name, config,target_domain,net_set)



=======
import os


def split_dset_structure_model_source_target(file_name, net_set):
    file_name = file_name.split(".mat")[0]
    target = file_name[-1]
    source = file_name[-3]
    file_name = file_name[:-4]
    dset = file_name.split("_")[0]
    r = len(dset)
    model_name = file_name[r:]

    model_structure = model_name.split("_")[0]
    return dset, model_structure, model_name, source, target


def extract_targetdomain_features(
    folder_path, dataset_name, config, target_domain, net_set, save_path
):

    files = sorted(os.listdir(folder_path))

    target_files = [
        file
        for file in files
        if f"_{target_domain}.mat" in file
        and file.split("_")[0] == dataset_name
        and file[-7] != target_domain
    ]

    if not os.path.exists(f"{save_path}/{dataset_name}_{config}"):
        os.makedirs(f"{save_path}/{dataset_name}_{config}")

    output_filename = f"{save_path}/{dataset_name}_{config}/{target_domain}.txt"
    with open(output_filename, "w") as output_file:
        for file_name in target_files:
            dset, model_structure, model_name, source, target = (
                split_dset_structure_model_source_target(file_name, net_set)
            )
            if model_name not in net_set:
                continue
            output_file.write(file_name + "\n")

    print(f"saved in {output_filename}")


if __name__ == "__main__":
    save_path = "./configs/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    folder_path = ""  # the feature_save_dir in train_source.py
    dataset_name = "office-home"

    net_set = [
        "_resnet50",
        "_resnet101",
        "_efficientnet_v2_l",
        "_efficientnet_v2_m",
        "_efficientnet_v2_s",
        "_swin_b",
        "_swin_l",
        "_swin_s",
        "_swin_t",
        "_vit_b_16",
        "_vit_b_32",
        "_vit_l_16",
        "_vit_l_32",
        "_vit_h_14",
        "_swin_v2_t",
        "_swin_v2_s",
        "_swin_v2_b",
    ]
    config = "main"
    for target_domain in range(4):
        target_domain = str(target_domain)
        extract_targetdomain_features(
            folder_path, dataset_name, config, target_domain, net_set, save_path
        )

    # config='s1'
    # net_set=['_resnet50_1e-05','_resnet50_0.0001','_resnet50_0.001','_resnet50_0.003','_resnet50_0.01','_resnet50_0.03','_resnet50_0.1']
    # for target_domain in range(4):
    #     target_domain=str(target_domain)
    #     extract_targetdomain_features(folder_path,dataset_name, config,target_domain,net_set)

    # config='s2'
    # net_set=['_resnet50_16','_resnet50_32','_resnet50_128','_resnet50_64','_resnet50_256']
    # for target_domain in range(4):
    #     target_domain=str(target_domain)
    #     extract_targetdomain_features(folder_path,dataset_name, config,target_domain,net_set)

    # config='s3'
    # net_set=['_resnet50_Adam','_resnet50_AdamW','_resnet50_ASGD','_resnet50']
    # for target_domain in range(4):
    #     target_domain=str(target_domain)
    #     extract_targetdomain_features(folder_path,dataset_name, config,target_domain,net_set)

    # config='s4'
    # net_set=['_resnet50_v1','_resnet50_nopre','_resnet50']
    # for target_domain in range(4):
    #     target_domain=str(target_domain)
    #     extract_targetdomain_features(folder_path,dataset_name, config,target_domain,net_set)
>>>>>>> 3b137b1 (Initial commit)
