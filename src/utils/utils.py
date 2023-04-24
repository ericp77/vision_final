import os

import torch
import torchvision.transforms as T


def load_parameters(file_weight: str, model: torch.nn.Module):
    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print(' size? ' + name, param.size(), model_dict[name].size())
            else:
                print(' name? ' + name)

        print(' loaded')
    else:
        print('weight file?')


def transform(img: torch.Tensor):
    # resize 256x256
    resize = T.Resize((256, 256))
    img = resize(img)

    # random crop 224x224
    crop = T.RandomCrop((224, 224))
    img = crop(img)

    # normalize
    normalize = T.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
    img = normalize(img.float())

    # permute
    img = torch.permute(img, (1, 0, 2, 3))
    return img


def get_class_name(idx: int):
    classes = [
        'brush_hair',
        'cartwheel',
        'catch',
        'chew',
        'clap',
        'climb',
        'climb_stairs',
        'dive',
        'draw_sword',
        'dribble',
        'drink',
        'eat',
        'fall_floor',
        'fencing',
        'flic_flac',
        'golf',
        'handstand',
        'hit',
        'hug',
        'jump',
        'kick',
        'kick_ball',
        'kiss',
        'laugh',
        'pick',
        'pour',
        'pullup',
        'punch',
        'push',
        'pushup',
        'ride_bike',
        'ride_horse',
        'run',
        'shake_hands',
        'shoot_ball',
        'shoot_bow',
        'shoot_gun',
        'sit',
        'situp',
        'smile',
        'smoke',
        'somersault',
        'stand',
        'swing_baseball',
        'sword',
        'sword_exercise',
        'talk',
        'throw',
        'turn',
        'walk',
        'wave'
    ]

    return classes[idx]
