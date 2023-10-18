model_name = "embedding"
weight = "./results/DermaMNIST/moco/checkpoint_0999.pth.tar"
model_type = "resnet18"
# model_type = "resnet18_cifar"
num_cluster = 7
batch_size = 1000
world_size = 1
workers = 4
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = 0
multiprocessing_distributed = True

data_test = dict(
    type="DermaMNIST",
    root_folder="./datasets/cifar10",
    embedding=None,
    train=True,
    all=False,
    split="train+test",
    shuffle=False,
    ims_per_batch=50,
    aspect_ratio_grouping=False,
    show=False,
    trans1=dict(
        aug_type="test",
        normalize=dict(mean=[0.5, 0.5, 0.5],
                       std=[0.5, 0.5, 0.5]),
    ),
    trans2=dict(
        aug_type="test",
        normalize=dict(mean=[0.5, 0.5, 0.5],
                       std=[0.5, 0.5, 0.5]),
    ),

)

model_sim = dict(
    type=model_type,
    num_classes=128,
    in_channels=3,
    in_size=28,
    batchnorm_track=True,
    test=False,
    feature_only=True,
    pretrained=weight,
    model_type="moco_embedding",
)


results = dict(
    output_dir="./results/DermaMNIST/{}".format(model_name),
)