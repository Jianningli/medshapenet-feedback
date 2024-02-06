import MedShapeNetCore
import argparse
import requests
import os
from clint.textui import progress
import sys
from glob import glob
import shutil
from urllib.request import urlopen
import re
from tqdm import tqdm
from pathlib import Path

class DevNull:
    def write(self, msg):
        pass

data_set_info={
    'homepage':'https://github.com/Jianningli/medshapenet-feedback/',
    'contact':'Jianning Li, jianningli.me@gmail.com',
    'version': f'MedShapeNetCore v{MedShapeNetCore.__version__}',
    'dataset': {               
               'ASOCA':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_ASOCA.npz?download=1',
                        'size':'41.8Mb',
                        'num_sample':'40',
                        'disease':'coronary artery disease,CAD,atherosclerosis,plaque',
                        'link':'https://asoca.grand-challenge.org/',
                        'information': 'coronary arteries',
                        'organs_key_words':'coronary arteries,artery',
                        'avi_keys':[
                                    'mask',
                                    'point',
                                    'mesh->vertices->sample index',
                                    'mesh->faces->sample index',
                                    'label'
                                    ]
                        },


               'FLARE':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_FLARE.npz?download=1',
                        'size':'555Mb',
                        'num_sample':'650',
                        'disease':' ',
                        'link':'https://flare.grand-challenge.org/',
                        'information': 'abdominal organs',
                        'organs_key_words':  'liver,kidney,spleen,pancreas,aorta,inferior vena cava,adrenal gland,gallbladder,esophagus,stomach,duodenum',
                        'avi_keys':[
                                    'organ->mask',
                                    'organ->point',
                                    'organ->mesh->vertices->sample index',
                                    'organ->mesh->faces->sample index'
                                    ]
                        },


               'KITS':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_KITS.npz?download=1',
                        'size':'401Mb',
                        'num_sample':'489',
                        'disease':'tumor,tumour',
                        'link':'https://kits-challenge.org/kits23/',
                        'information': 'kidney and kidney tumor',
                        'organs_key_words':'kidney,tumor',
                        'avi_keys':[
                                    'mask',
                                    'point',
                                    'mesh->vertices->sample index',
                                    'mesh->faces->sample index',
                                    'label->mask'
                                    ]
                        },


               'PULMONARY':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_PULMONARY.npz?download=1',
                        'size':'1.14Gb',
                        'num_sample':'2397',
                        'disease':' ',
                        'link':'https://arxiv.org/pdf/2309.17329.pdf',
                        'information': 'pulmonary arteries, including the airway,artery, vein',
                        'organs_key_words':'pulmonary arteries, airway,artery,vein,lung',
                        'avi_keys':[
                                    'organ->mask',
                                    'orgna->point',
                                    'organ->mesh->vertices->sample index',
                                    'organ->mesh->faces->sample index'
                                    ]
                        },


               'ThoracicAorta_Saitta':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_ThoracicAorta_Saitta.npz?download=1',
                        'size':'515.57Mb',
                        'num_sample':'500',
                        'disease':'thoracic aortic aneurysm',
                        'link':'https://pubmed.ncbi.nlm.nih.gov/35083618/',
                        'information': 'thoracic aorta with arch branches',
                        'organs_key_words':'thoracic aorta',
                        'avi_keys':[
                                    'mask->sample index',
                                    'point->sample index',
                                    'mesh->vertices->sample index',
                                    'mesh->faces->sample index'
                                    ]
                        },


               'CoronaryArteries':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_CoronaryArteries.npz?download=1',
                        'size':'677.02Mb',
                        'num_sample':'544',
                        'disease':'coronary artery calcium,CAC,tortuosity',
                        'link':'https://pubs.aip.org/aip/apb/article/8/1/016103/3061557/A-fully-automated-deep-learning-approach-for',
                        'information': 'coronary arteries',
                        'organs_key_words':'coronary arteries, artery',
                        'avi_keys':[
                                    'mask',
                                    'point',
                                    'mesh->vertices->sample index',
                                    'mesh->faces->sample index'
                                    ]
                        },

               '3DTeethSeg':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_3DTeethSeg.npz?download=1',
                        'size':'3.7Gb',
                        'num_sample':'1800',
                        'disease':' ',
                        'link':'https://github.com/abenhamadou/3DTeethSeg22_challenge',
                        'information': '3D teeth labeling and semantic segmentation',
                        'organs_key_words':'teeth,tooth',
                        'avi_keys':[
                                    'patient ID->mesh->vertices',
                                    'patient ID->mesh->faces'
                                    ]
                        },

               'FaceVR':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_FaceVR.npz?download=1',
                        'size':'14.9Mb',
                        'num_sample':'11',
                        'disease':' ',
                        'link':'https://figshare.com/articles/dataset/Medical_Augmented_Reality_Facial_Data_Collection/8857007/2',
                        'information': '3D facial models for VR',
                        'organs_key_words':'face,facial',
                        'avi_keys':[
                                    'point',
                                    'mesh->vertices',
                                    'mesh->faces'
                                    ]
                        },

               'ToothFairy':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_ToothFairy.npz?download=1',
                        'size':'145.94Mb',
                        'num_sample':'153',
                        'disease':' ',
                        'link':'https://toothfairychallenges.github.io/',
                        'information': 'inferior alveolar nerve segmentations',
                        'organs_key_words':'maxillofacial,teeth,tooth,inferior alveolar nerve',
                        'avi_keys':[
                                    'mask->sample index',
                                    'point->sample index',
                                    'mesh->vertices->sample index',
                                    'mesh->faces->sample index'
                                    ]
                        },

               'AutoImplantCraniotomy':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_AutoImplantCraniotomy.npz?download=1',
                        'size':'6.41Mb',
                        'num_sample':'11',
                        'disease':'cranial defect,craniotomy',
                        'link':'https://autoimplant2021.grand-challenge.org/',
                        'information': 'skull reconstruction and cranial implant design',
                        'organs_key_words':'skull,craniotomy',
                        'avi_keys':[
                                    'mask'
                                    ]
                        },

               'AVT':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_AVT.npz?download=1',
                        'size':'115.37Mb',
                        'num_sample':'42',
                        'disease':' ',
                        'link':'https://pubmed.ncbi.nlm.nih.gov/35059483/',
                        'information': 'aortic vessel tree',
                        'organs_key_words':'aorta,vessel,vessels,tree',
                        'avi_keys':[
                                    'mask->sample index',
                                    'point->sample index',
                                    'mesh->vertices->sample index',
                                    'mesh->faces->sample index'
                                    ]
                        },

               'SurgicalInstruments':{'url':'https://zenodo.org/records/10623506/files/medshapenetcore_SurgicalInstruments.npz?download=1',
                        'size':'5.30Mb',
                        'num_sample':'106',
                        'disease':' ',
                        'link':'https://www.nature.com/articles/s41597-023-02684-0',
                        'information': '3D surgical instrument models',
                        'organs_key_words':'surgical, surgery, instruments, instrument',
                        'avi_keys':[
                                    'mask->sample index',
                                    'point->sample index',
                                    'mesh->vertices->sample index',
                                    'mesh->faces->sample index'
                                    ]
                        }

                },

    'commands': [
                 'python -m MedShapeNetCore download DARASET',
                 'python -m MedShapeNetCore clean',
                 'python -m MedShapeNetCore check_available_keys DARASET',
                 'python -m MedShapeNetCore search_by_organ ORGAN',
                 'python -m MedShapeNetCore search_by_disease DISEASE',
                 'python -m MedShapeNetCore search_and_download ORGAN'                
    ]

}

    
def search_by_organ():
    
    """
    Search the MedShapeNetCore database using 
    organ keywords, such as liver. Abbrevations
    are supported.
    """

    organ=sys.argv[2] 
    datasets=list(data_set_info['dataset'].keys())
    get_datasets=[]
    for data_set in datasets:
        if organ in data_set_info['dataset'][data_set]['organs_key_words']:
            get_datasets.append(data_set)

    if len(get_datasets)!=0:
        print(
            f'the following dataset(s): {get_datasets} contain(s) {organ}  \n'
            f'download {get_datasets} using the following command:         \n'
            f'python -m MedShapeNetCore download {get_datasets}'
            )

    else:
        print(
            f'{organ} not found in the MedShapeNetCore  \n'
            'current availale anatomy keywords are:             \n'
            '_____________'
            )

        for data_set in datasets:
            print(data_set_info['dataset'][data_set]['organs_key_words'])
    print('______function description______:')
    help(search_by_organ)





def search_by_disease():

    """
    Search the MedShapeNetCore database using
    disease keywords, such as tumor. Abbrevations
    are supported.
    """
    disease=sys.argv[2] 
    datasets=list(data_set_info['dataset'].keys())
    get_datasets=[]
    for data_set in datasets:
        if disease in data_set_info['dataset'][data_set]['disease']:
            get_datasets.append(data_set)

    if len(get_datasets)!=0:
        print(
            f'the following dataset(s): {get_datasets} contain(s) {disease}  \n'
            f'download {get_datasets} using the following command:         \n'
            f'python -m MedShapeNetCore download {get_datasets}'
            )

    else:
        print(
            f'{disease} not found in the MedShapeNetCore  \n'
            'current availale diseases are:             \n'
            '_____________'
            )

        for data_set in datasets:
            print(data_set_info['dataset'][data_set]['disease'])
    print('______function description______:')
    help(search_by_disease)


def search_and_download():

    """
    Search the online MedShapeNet database  
    using organ nomenclature, such as liver,
    and download the corresponding .stl files
    """

    if not os.path.exists('./temp/'):
        os.mkdir('./temp/')
    r = requests.get("https://medshapenet.ikim.nrw/uploads/MedShapeNetDataset.txt", stream=True)
    with open('./temp/MedShapeNetDataset.txt', 'wb') as f:
        f.write(r.content)
    print(f'searching {sys.argv[2]}...')
    matched_urls=[]
    with open('./temp/MedShapeNetDataset.txt', 'r') as inF:
        for line in inF:
            if sys.argv[2] in line:
                matched_urls.append(line)
    if len(matched_urls)==0:
        print(f'found {len(matched_urls)} entries of {sys.argv[2]}')
        if os.path.exists('./temp/'):
            shutil.rmtree('./temp/')
    else:
        save_folder='./stls/'+sys.argv[2]+'/'
        if not os.path.exists(save_folder):
            Path(save_folder).mkdir(parents=True, exist_ok=True)
        print(f'found {len(matched_urls)} entries of {sys.argv[2]}, started downloading... files are saved in folder {save_folder}')
        counter=0
        print('_________ urls:')
        for url in matched_urls:
            print(url)
            r = requests.get(url.strip(), stream=True)
            filename=save_folder+sys.argv[2]+'_'+'{0:05}'.format(counter)+'.stl'
            counter+=1
            with open(filename, 'wb') as f:
                f.write(r.content)
        print(f'Download complete! Files are stored in folder {save_folder}')
        if os.path.exists('./temp/'):
            shutil.rmtree('./temp/')

    print('______function description______:')
    help(search_and_download)


def info():

    """
    Display the general information about MedShapeNetCore,
    including the currently available commands, datasets and
    their descriptions, and the number of samples
    """

    print('___GeneralInfo___')

    print(f'MedShapeNetCore v{MedShapeNetCore.__version__}')
    print(f'Homepage:', data_set_info['homepage'])
    print('Contact:', data_set_info['contact'])

    print('___available datasets___')

    ASOCA=data_set_info['dataset']['ASOCA']
    FLARE=data_set_info['dataset']['FLARE']
    KITS=data_set_info['dataset']['KITS']
    PULMONARY=data_set_info['dataset']['PULMONARY']
    ThoracicAorta_Saitta=data_set_info['dataset']['ThoracicAorta_Saitta']
    CoronaryArteries=data_set_info['dataset']['CoronaryArteries']
    TDTeethSeg=data_set_info['dataset']['3DTeethSeg']
    FaceVR=data_set_info['dataset']['FaceVR']
    ToothFairy=data_set_info['dataset']['ToothFairy']
    AutoImplantCraniotomy=data_set_info['dataset']['AutoImplantCraniotomy']
    SurgicalInstruments=data_set_info['dataset']['SurgicalInstruments'] 
    AVT=data_set_info['dataset']['AVT'] 
    print(
           f'ASOCA:                 {ASOCA}                     \n'
           '___\n'
           f'FLARE:                 {FLARE}                     \n'
           '___\n'
           f'KITS:                  {KITS}                      \n'
           '___\n'
           f'PULMONARY:             {PULMONARY}                 \n'
           '___\n'
           f'ThoracicAorta_Saitta:  {ThoracicAorta_Saitta}      \n'
           '___\n'
           f'CoronaryArteries:      {CoronaryArteries}          \n'
           '___\n'
           f'FaceVR:                {FaceVR}                    \n'
           '___\n'

           f'ToothFairy:            {ToothFairy}                \n'
           '___\n'

           f'AutoImplantCraniotomy: {AutoImplantCraniotomy}     \n'
           '___\n'

           f'SurgicalInstruments:   {SurgicalInstruments}        \n'
           '___\n'
           f'AVT:                   {AVT}                        \n'
           '___\n'
           f'3DTeethSeg:            {TDTeethSeg}                  '

           )

    counter=0
    for keys in list(data_set_info['dataset'].keys()):
        counter+=int(data_set_info['dataset'][keys]['num_sample'])

    print('_____')
    print('number of datasets:',len(list(list(data_set_info['dataset'].keys()))))
    print('total number of samples:', counter)


    print('___basic commands___')

    print(data_set_info['commands'])

    print('______function description______:')
    help(info)


def download():

    """
    Download the .npz file(s) to be used in Python.
    """

    if not os.path.exists('./medshapenetcore_npz/'):
        os.mkdir('./medshapenetcore_npz/')
    available_datasets=list(data_set_info['dataset'].keys())
    if sys.argv[2] == 'all':
        print('downloading all available datasets:',available_datasets)
        print('Warning:this may take a long time depending on your internet connection!')
        os.system("zenodo_get %s"%'10.5281/zenodo.10406279 -o medshapenetcore_npz')
    else:
        if not (sys.argv[2] in available_datasets):
            print(
                f"dataset {sys.argv[2]} not available," +
                f"please choose from {available_datasets}"
                )

            raise Exception()
        else:
            if sys.argv[2] == 'ASOCA':
                url=data_set_info['dataset']['ASOCA']['url']
                path = 'medshapenetcore_ASOCA.npz'

            if sys.argv[2] == 'FLARE':
                url=data_set_info['dataset']['FLARE']['url']
                path = 'medshapenetcore_FLARE.npz'

            if sys.argv[2]== 'KITS':
                url=data_set_info['dataset']['KITS']['url']
                path = 'medshapenetcore_KITS.npz'

            if sys.argv[2]== 'PULMONARY':
                url=data_set_info['dataset']['PULMONARY']['url']
                path = 'medshapenetcore_PULMONARY.npz'

            if sys.argv[2]== 'ThoracicAorta_Saitta':
                url=data_set_info['dataset']['ThoracicAorta_Saitta']['url']
                path = 'medshapenetcore_ThoracicAorta_Saitta.npz'

            if sys.argv[2]== 'CoronaryArteries':
                url=data_set_info['dataset']['CoronaryArteries']['url']
                path = 'medshapenetcore_CoronaryArteries.npz'

            if sys.argv[2]== '3DTeethSeg':
                url=data_set_info['dataset']['3DTeethSeg']['url']
                path = 'medshapenetcore_3DTeethSeg.npz'

            if sys.argv[2]== 'FaceVR':
                url=data_set_info['dataset']['FaceVR']['url']
                path = 'medshapenetcore_FaceVR.npz'

            if sys.argv[2]== 'AutoImplantCraniotomy':
                url=data_set_info['dataset']['AutoImplantCraniotomy']['url']
                path = 'medshapenetcore_AutoImplantCraniotomy.npz'

            if sys.argv[2]== 'ToothFairy':
                url=data_set_info['dataset']['ToothFairy']['url']
                path = 'medshapenetcore_ToothFairy.npz'

            if sys.argv[2]== 'SurgicalInstruments':
                url=data_set_info['dataset']['SurgicalInstruments']['url']
                path = 'medshapenetcore_SurgicalInstruments.npz'

            if sys.argv[2]== 'AVT':
                url=data_set_info['dataset']['AVT']['url']
                path = 'medshapenetcore_AVT.npz'


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
                print(
                       'Download went wrong! ' + 
                       'Please download the dataset manually at: ' +
                       'https://zenodo.org/records/10423181' +
                       'and copy it to folder medshapenetcore_npz'+
                       'inside the current working directory'
                       )

                raise RuntimeError()
    print('______function description______:')
    help(download)



def clean():
    """
    Deleting all files generated by MedShapeNet, including
    temporary files and the downloaded .npz files.
    """
    print('deleting all files generated by MedShapeNet...')
    if os.path.exists('./medshapenetcore_npz/'):
        shutil.rmtree('./medshapenetcore_npz/')

    if os.path.exists('./medshapenetcore_saved/'):
        shutil.rmtree('./medshapenetcore_saved/')
    print('______function description______:')
    help(clean)


def check_available_keys():
    """
    Check the available keys in each dataset formatted as .npz
    files. 
    """

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
        print('___AvailableKeys___')
        print(data_set_info['dataset']['PULMONARY']['avi_keys'])
        
    if sys.argv[2] == 'ThoracicAorta_Saitta':
        print('___AvailableKeys___')
        print(data_set_info['dataset']['ThoracicAorta_Saitta']['avi_keys'])

    if sys.argv[2] == 'CoronaryArteries':
        print('___AvailableKeys___')
        print(data_set_info['dataset']['CoronaryArteries']['avi_keys'])

    if sys.argv[2] == '3DTeethSeg':
        print('___AvailableKeys___')
        print(data_set_info['dataset']['3DTeethSeg']['avi_keys'])

    if sys.argv[2] == 'FaceVR':
        print('___AvailableKeys___')
        print(data_set_info['dataset']['FaceVR']['avi_keys'])
        
    if sys.argv[2] == 'AutoImplantCraniotomy':
        print('___AvailableKeys___')
        print(data_set_info['dataset']['AutoImplantCraniotomy']['avi_keys'])

    if sys.argv[2] == 'ToothFairy':  
        print('___AvailableKeys___')
        print(data_set_info['dataset']['ToothFairy']['avi_keys'])

    if sys.argv[2] == 'SurgicalInstruments': 
        print('___AvailableKeys___')
        print(data_set_info['dataset']['SurgicalInstruments']['avi_keys'])

    if sys.argv[2] == 'AVT': 
        print('___AvailableKeys___')
        print(data_set_info['dataset']['AVT']['avi_keys'])

    print('______function description______:')
    help(check_available_keys)



if __name__ == "__main__":
    import fire
    sys.stderr = DevNull()
    fire.Fire()
 




