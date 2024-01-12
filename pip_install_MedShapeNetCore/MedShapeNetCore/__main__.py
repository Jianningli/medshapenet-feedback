import MedShapeNetCore
import argparse
import requests
import os
from clint.textui import progress
import sys
from glob import glob
import shutil

class DevNull:
    def write(self, msg):
        pass

data_set_info={
    'homepage':'https://github.com/Jianningli/medshapenet-feedback/',
    'contact':'Jianning Li, jianningli.me@gmail.com',
    'version': f'MedShapeNetCore v{MedShapeNetCore.__version__}',
    'dataset': {               
               'ASOCA':{'url':'https://zenodo.org/records/10423181/files/medshapenetcore_ASOCA.npz?download=1',
                        'size':'41.8Mb',
                        'link':'https://asoca.grand-challenge.org/',
                        'information': 'coronary arteries',
                        'avi_keys':[
                                    'mask',
                                    'point',
                                    'mesh->vertices->sample index',
                                    'mesh->faces->sample index',
                                    'label'
                                    ]
                        },


               'FLARE':{'url':'https://zenodo.org/records/10423181/files/medshapenetcore_FLARE.npz?download=1',
                        'size':'555Mb',
                        'link':'https://flare.grand-challenge.org/',
                        'information': 'abdominal organs',
                        'avi_keys':[
                                    'organ->mask',
                                    'orgna->point',
                                    'organ->mesh->vertices->sample index',
                                    'organ->mesh->faces->sample index'
                                    ]
                        },


               'KITS':{'url':'https://zenodo.org/records/10423181/files/medshapenetcore_KITS.npz?download=1',
                        'size':'401Mb',
                        'link':'https://kits-challenge.org/kits23/',
                        'information': 'kidney and kidney tumor',
                        'avi_keys':[
                                    'mask',
                                    'point',
                                    'mesh->vertices->sample index',
                                    'mesh->faces->sample index',
                                    'label->mask'
                                    ]
                        },


               'PULMONARY':{'url':'https://zenodo.org/records/10423181/files/medshapenetcore_PULMONARY.npz?download=1',
                        'size':'1.14Gb',
                        'link':'https://arxiv.org/pdf/2309.17329.pdf ',
                        'information': 'pulmonary arteries, including the airway,artery, vein',
                        'avi_keys':[
                                    'organ->mask',
                                    'orgna->point',
                                    'organ->mesh->vertices->sample index',
                                    'organ->mesh->faces->sample index'
                                    ]
                        }
                },

    'commands': [
                 'python -m MedShapeNetCore download DARASET',
                 'python -m MedShapeNetCore clean',
                 'python -m MedShapeNetCore check_available_keys DARASET'
    ]

}
    


def info():
    print('______copyright')

    print(f'MedShapeNetCore v{MedShapeNetCore.__version__}')
    print(f'Homepage:', data_set_info['homepage'])
    print('Contact:', data_set_info['contact'])

    print('______available datasets')

    ASOCA=data_set_info['dataset']['ASOCA']
    FLARE=data_set_info['dataset']['FLARE']
    KITS=data_set_info['dataset']['KITS']
    PULMONARY=data_set_info['dataset']['PULMONARY']

    print(
           f'ASOCA:     {ASOCA}      \n'
           '___\n'
           f'FLARE:     {FLARE}      \n'
           '___\n'
           f'KITS:      {KITS}       \n'
           '___\n'
           f'PULMONARY: {PULMONARY}    '
            )

    print('______basic commands')

    print(data_set_info['commands'])



def download():

    if not os.path.exists('./medshapenetcore_npz/'):
        os.mkdir('./medshapenetcore_npz/')

    available_datasets=list(data_set_info['dataset'].keys())

    if not (sys.argv[2] in available_datasets):
        print(
            f"dataset {sys.argv[2]} not available," +
            f"please choose from {available_datasets}"
            )

        raise Exception()

    elif sys.argv[2] == 'ASOCA':
        url=data_set_info['dataset']['ASOCA']['url']
        path = 'medshapenetcore_ASOCA.npz'

    elif sys.argv[2] == 'FLARE':
        url=data_set_info['dataset']['FLARE']['url']
        path = 'medshapenetcore_FLARE.npz'

    elif sys.argv[2]== 'KITS':
        url=data_set_info['dataset']['KITS']['url']
        path = 'medshapenetcore_KITS.npz'

    elif sys.argv[2]== 'PULMONARY':
        url=data_set_info['dataset']['PULMONARY']['url']
        path = 'medshapenetcore_PULMONARY.npz'

    '''
    elif sys.argv[2] == 'ALL':
        url='https://zenodo.org/records/10423181/files/medshapenetcore.npz?download=1'
        path = './medshapenetcore_npz/medshapenetcore.npz'
    '''

    print('downloading...')

    save_dir='./medshapenetcore_npz/'+path

    r = requests.get(url, stream=True)
    with open(save_dir, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
            if chunk:
                f.write(chunk)
                f.flush()

    if os.path.exists(save_dir):
        print('download complete...')
        print('file directory:',save_dir)
    else:
        raise RuntimeError(
                           'Download went wrong! ' + 
                           'Please download the dataset manually at: ' +
                           'https://zenodo.org/records/10423181' +
                           'and copy it to folder medshapenetcore_npz'
                           )



def clean():
    print('deleting all files generated by MedShapeNet...')
    if os.path.exists('./medshapenetcore_npz/'):
        shutil.rmtree('./medshapenetcore_npz/')

    if os.path.exists('./medshapenetcore_saved/'):
        shutil.rmtree('./medshapenetcore_saved/')




def check_available_keys():
    available_datasets=list(data_set_info['dataset'].keys())
    if not (sys.argv[2] in available_datasets):

        print(
            f"dataset {sys.argv[2]} not available," +
            f"please choose from {available_datasets}"
            )

        raise Exception()

    print(
         '___size___\n'
         'mask: Mx(LxWxH) \n'
         'point: MxNx3  \n'
         'mesh->vertices->sample index:  Nvx3  \n'
         'mesh->faces->sample index:  Nfx3  \n'
         'label:Mx1, heathy (0), pathological (1) \n'
         'label->mask:Mx(LxWxH) \n'

         '___notation___\n'
         'M:  the  number of samples\n'
         'sample index: integer from 0 to M  \n'
         'N:  the number of points \n'
         'N:  the number of points \n'
         'Nv: the number of vertices of the sample  \n'
         'Nf: the number of faces of the sample \n'
        )

    if sys.argv[2] == 'ASOCA':
        print('___AvailableKeys___')
        print(data_set_info['dataset']['ASOCA']['avi_keys'])


    if sys.argv[2] == 'FLARE':
        flare_organs=[
                     'liver',
                     'right_kidney',
                     'spleen',
                     'pancreas',
                     'aorta',
                     'inferior_vena_cava',
                     'right_adrenal_gland', 
                     'left_adrenal_gland',
                     'gallbladder',
                     'esophagus',
                     'stomach',
                     'duodenum',
                     'left_kidney'
                     ]
        print(f'organ keys: {flare_organs}')
        print('___AvailableKeys___')
        print(data_set_info['dataset']['FLARE']['avi_keys'])



    if sys.argv[2] == 'PULMONARY':
        PULMONARY_organs=[
                         'airway',
                         'artery',
                         'vein'
                         ]
        print( f'organ keys: {PULMONARY_organs}')
        print('___AvailableKeys___')
        print(data_set_info['dataset']['PULMONARY']['avi_keys'])


    if sys.argv[2] == 'KITS':
        print(data_set_info['dataset']['PULMONARY']['avi_keys'])
        

    '''    
    if sys.argv[2] == 'ALL':
        print('___')
        print(
               f'add an addition key {list(data_set_info['dataset'].keys())} before the keys of the dataset         \n'
                )
        print('___')
    '''

if __name__ == "__main__":
    import fire
    sys.stderr = DevNull()
    fire.Fire()
 




