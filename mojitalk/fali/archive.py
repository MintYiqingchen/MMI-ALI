import os
_home = os.getenv('HOME')
archive = {
    'image':{
        'cityscape':{
            'train':[
                _home + '/datasets/cityscape/train/[a-s]*/*.png', 
                _home + '/datasets/cityscape_gtFine/gtFine/train/[a-s]*/*_color.png',
                _home + '/datasets/RAND_CITYSCAPES/RGB/000[01]*.png'
                ],
            'val':[
                _home + '/datasets/cityscape/val/*/*', 
                _home + '/datasets/cityscape_gtFine/gtFine/val/*/*_color.png',
                _home + '/datasets/RAND_CITYSCAPES/RGB/0002[0-4]*.png'
                ],
            'test':[
                _home + '/datasets/cityscape/train/[t-z]*/*.png',
                _home + '/datasets/RAND_CITYSCAPES/RGB/0003[0-2]*.png', # '/datasets/cityscape_gtFine/gtFine/train/[t-z]*/*_color.png',
                _home + '/datasets/RAND_CITYSCAPES/RGB/0003[0-2]*.png'
                ],
            'sup': [(1,), (0,), set()]
        },
        'animal':{
            'train':[
                _home+ '/codework/CycleGAN-tensorflow/datasets/horse2zebra/trainA/*',
                _home+ '/codework/CycleGAN-tensorflow/datasets/horse2zebra/trainB/*'
                ],
            'val':[
                _home+ '/codework/CycleGAN-tensorflow/datasets/horse2zebra/testA/*',
                _home+ '/codework/CycleGAN-tensorflow/datasets/horse2zebra/testB/*'
                ],
            'test':[
                _home+ '/codework/CycleGAN-tensorflow/datasets/horse2zebra/testA/*',
                _home+ '/codework/CycleGAN-tensorflow/datasets/horse2zebra/testB/*'
                ]
        },
        'paint':{
            'train':[
                _home + '/datasets/style_transfer/monet/*',
                _home + '/datasets/style_transfer/photo/*',
                _home + '/datasets/style_transfer/ukiyoe/*'
                ],
            'val' :[
                _home + '/datasets/style_transfer/monet_test/*',
                _home + '/datasets/style_transfer/photo_test/*',
                _home + '/datasets/style_transfer/ukiyoe_test/*'
                ],
            'test':[
                _home + '/datasets/style_transfer/monet_test/*',
                _home + '/datasets/style_transfer/photo_test/*',
                _home + '/datasets/style_transfer/ukiyoe_test/*'
                ]
        }
    },
    'text' :{
        'mojitalk':{
            'embed': {
                'train': [
                    _home+'/datasets/mojitalk_data/fali_ver/train_HAPPY.npy',
                    _home+'/datasets/mojitalk_data/fali_ver/train_ANGRY.npy',
                    _home+'/datasets/mojitalk_data/fali_ver/train_PENSIVE.npy',
                    _home+'/datasets/mojitalk_data/fali_ver/train_ABASH.npy',
                    ],
                'val': [
                    _home+'/datasets/mojitalk_data/fali_ver/dev_HAPPY.npy',
                    _home+'/datasets/mojitalk_data/fali_ver/dev_ANGRY.npy',
                    _home+'/datasets/mojitalk_data/fali_ver/dev_PENSIVE.npy',
                    _home+'/datasets/mojitalk_data/fali_ver/dev_ABASH.npy',
                    ],
                'test': [
                    _home+'/datasets/mojitalk_data/fali_ver/test_HAPPY.npy',
                    _home+'/datasets/mojitalk_data/fali_ver/test_ANGRY.npy',
                    _home+'/datasets/mojitalk_data/fali_ver/test_PENSIVE.npy',
                    _home+'/datasets/mojitalk_data/fali_ver/test_ABASH.npy',
                    ]
                },
            'text': {
                'train': [
                    _home+'/datasets/mojitalk_data/fali_ver/train_HAPPY.txt',
                    _home+'/datasets/mojitalk_data/fali_ver/train_ANGRY.txt',
                    _home+'/datasets/mojitalk_data/fali_ver/train_PENSIVE.txt',
                    _home+'/datasets/mojitalk_data/fali_ver/train_ABASH.txt',
                    ],
                'val': [
                    _home+'/datasets/mojitalk_data/fali_ver/dev_HAPPY.txt',
                    _home+'/datasets/mojitalk_data/fali_ver/dev_ANGRY.txt',
                    _home+'/datasets/mojitalk_data/fali_ver/dev_PENSIVE.txt',
                    _home+'/datasets/mojitalk_data/fali_ver/dev_ABASH.txt',
                    ],
                'test': [
                    _home+'/datasets/mojitalk_data/fali_ver/test_HAPPY.txt',
                    _home+'/datasets/mojitalk_data/fali_ver/test_ANGRY.txt',
                    _home+'/datasets/mojitalk_data/fali_ver/test_PENSIVE.txt',
                    _home+'/datasets/mojitalk_data/fali_ver/test_ABASH.txt',
                    ]
                }
            }
    }
}
