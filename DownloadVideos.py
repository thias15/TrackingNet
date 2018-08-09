from __future__ import unicode_literals
from subprocess import check_call
from concurrent import futures

import argparse

import subprocess
from tqdm import tqdm

import youtube_dl
import socket

import os
import shutil
import zipfile

import io
import sys
import codecs

import cv2
import pandas as pd
import numpy as np
import csv
import itertools
import string

import xml.etree.ElementTree as ET


# if sys.stdout.encoding != 'UTF-8':
#     sys.stdout = codecs.getwriter('utf-8')(sys.stdout, 'strict')
# if sys.stderr.encoding != 'UTF-8':
#     sys.stderr = codecs.getwriter('utf-8')(sys.stderr, 'strict')


# Print iterations progress (thanks StackOverflow)
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    #bar             = '█' * filledLength + '-' * (barLength - filledLength)
    bar             = '0' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
    
def find_offset(array,value):
    offset = (np.abs(array-value)).min()
    return offset
    
def find_offset_right(array,value):
    offsets = np.array(array-value)
    offset = offsets[offsets >= 0].min()
    return offset

def is_zero_file(file_path):  
    return os.path.isfile(file_path) and os.path.getsize(file_path) < 1024

def has_zero_file(folder_path): 
    # if (not os.path.exists(folder_path)):
        # return True
    Res = [is_zero_file(os.path.join(folder_path,path)) for path in os.listdir(folder_path)]
    return (True in Res or len(Res) == 0)


# Download and cut a clip to size
def dl_and_cut(data, d_set_dir, img_dir, anno_dir, video_dir, zips_dir, bRemove = False, bSaveAnno = False, bExtractFrame = False, bZipFrame = False, bZeroPad = False, bCheckFiles=False, bTrainSet=False, bDrawBB = False):


    vid = data['youtube_id'].values[0]
    seg = data['segment_id'].values[0]
    
    #print(vid, seg)

    video_path = os.path.join(video_dir, vid + '.mp4')     
    frame_path = os.path.join(img_dir, seg)   
    img_extension = '.jpg'
    frame_bb_path = frame_path.replace('imgs', 'imgs_bb')
    zip_path = os.path.join(zips_dir, seg+'.zip')
    xml_path = os.path.join(anno_dir, seg + ".xml")
    otb_path = os.path.join(anno_dir.replace('anno', 'anno_otb'), seg + ".txt")     
    

    if(os.path.exists(video_path) and os.path.exists(frame_path)):
        if (bCheckFiles):
            if (has_zero_file(frame_path)):
                print("Replacing video due to bad frames:", vid)
                os.remove(video_path)

    
    if (not os.path.exists(video_path)):
        try:
            # Use youtube_dl to download the video if does not exist already
            ydl_opts = {'quiet':True, 'ignoreerrors':True, 'no_warnings':True,
                        'format': 'best[ext=mp4]',
                        'outtmpl': video_path}
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(['youtu.be/'+vid+'.mp4'])



        except Exception as e: 
            cnt_error = cnt_error + 1
            print("**************************************************")
            print(e)


    if not os.path.exists(video_path):
        print("ERROR: " + vid + ": Video not here")


    # Verify that the video has been downloaded. Skip otherwise
    if (os.path.exists(video_path)):
    
        if (os.path.exists(xml_path) and os.path.exists(otb_path)):
            bSaveAnno = False
    
    
        if (bSaveAnno or bExtractFrame or bZipFrame):
            capture = cv2.VideoCapture(video_path)
            fps, total_f = capture.get(5), capture.get(7)
            vid_w, vid_h = capture.get(3),capture.get(4)
            target_fps = fps
            time_step = 1/target_fps * 1000
            #print('time_step:', time_step)   
            
            labeled_timestamps = data['timestamp_ms'].values
            
            timestamps = np.arange(labeled_timestamps[0], labeled_timestamps[-1], time_step)
            nb_frames = len(timestamps)
            #print ('Number frames:',nb_frames)
            
            # Get nearest frame for every labeled timestamp
            labeled_indexes = []
            for label in labeled_timestamps:
                labeled_frame_index = find_nearest(timestamps, label)
                labeled_indexes.append(labeled_frame_index)
            #print(labeled_indexes)
        
        if (bSaveAnno):
            # create the file structure
            annotation = ET.Element('annotation')

            folder = ET.SubElement(annotation, 'folder')  
            filename = ET.SubElement(annotation, 'filename')        
            source = ET.SubElement(annotation, 'source')
            obj = ET.SubElement(annotation, 'object')  
            folder.text = 'Tracking-Net'  
            filename.text = seg 

            type = ET.SubElement(source, 'type')
            srcImg = ET.SubElement(source, 'sourceImage')   
            srcAnno = ET.SubElement(source, 'sourceAnnotation')          
            type.text = 'video' 
            srcImg.text = 'vatic frames' 
            srcAnno.text = 'vatic'   
            
            name = ET.SubElement(obj, 'name') 
            moving = ET.SubElement(obj, 'moving') 
            action = ET.SubElement(obj, 'action') 
            verified = ET.SubElement(obj, 'verified') 
            id = ET.SubElement(obj, 'id') 
            createdFrame = ET.SubElement(obj, 'createdFrame') 
            startFrame = ET.SubElement(obj, 'startFrame') 
            endFrame = ET.SubElement(obj, 'endFrame') 

            # fill the structure
            name.text = data['class_name'].values[0]   
            moving.text = 'true'     
            # action.text = '' 
            verified.text = '0'         
            id.text = str(data['class_id'].values[0])
            createdFrame.text = '0' 
            startFrame.text = '0' 
            endFrame.text = str(nb_frames-1)
            
            otb_rows = []


        if (bExtractFrame):
            if (os.path.exists(frame_path)):
                nb_files = len([fn for fn in os.listdir(frame_path) if fn.endswith(img_extension)])
                
                if (nb_frames != nb_files): # Missing frames
                    print("Replacing img and zip due to missing frames:", seg)
                    shutil.rmtree(frame_path)
                    os.makedirs(frame_path) 
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    if (os.path.exists(frame_bb_path)): #Also clean frame_bb_path
                        shutil.rmtree(frame_bb_path)
                        os.makedirs(frame_bb_path)                        
               
                elif (bCheckFiles):
                    if (has_zero_file(frame_path)): # Zero byte frames
                        print("Replacing img and zip due to bad frames:", seg)
                        shutil.rmtree(frame_path)
                        os.makedirs(frame_path) 
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        if (os.path.exists(frame_bb_path)): #Also clean frame_bb_path
                            shutil.rmtree(frame_bb_path)
                            os.makedirs(frame_bb_path)  
                    else:
                        bExtractFrame = False # Force not extracting frames
                            
                else:
                    bExtractFrame = False # Force not extracting frames
                
            else:
                os.makedirs(frame_path) 
                
            if (bDrawBB and not os.path.exists(frame_bb_path)):
                os.makedirs(frame_bb_path)

               
        if (bSaveAnno and not bTrainSet): #random initial bounding box
            my_l = 0
            xmin = 10
            xmax = 200
            ymin = 10
            ymax = 200
            # XML bounding box polygon
            polygon = ET.SubElement(obj, 'polygon') 
            t = ET.SubElement(polygon, 't') 
            t.text = str(0)
            pt = ET.SubElement(polygon, 'pt') 
            x = ET.SubElement(pt, 'x') 
            y = ET.SubElement(pt, 'y') 
            l = ET.SubElement(pt, 'l') 
            x.text = str(xmin)
            y.text = str(ymin)
            l.text = str(my_l)
            pt = ET.SubElement(polygon, 'pt') 
            x = ET.SubElement(pt, 'x') 
            y = ET.SubElement(pt, 'y') 
            l = ET.SubElement(pt, 'l') 
            x.text = str(xmin)
            y.text = str(ymax)
            l.text = str(my_l) 
            pt = ET.SubElement(polygon, 'pt') 
            x = ET.SubElement(pt, 'x') 
            y = ET.SubElement(pt, 'y') 
            l = ET.SubElement(pt, 'l') 
            x.text = str(xmax)
            y.text = str(ymax)
            l.text = str(my_l) 
            pt = ET.SubElement(polygon, 'pt') 
            x = ET.SubElement(pt, 'x') 
            y = ET.SubElement(pt, 'y') 
            l = ET.SubElement(pt, 'l') 
            x.text = str(xmax)
            y.text = str(ymin)
            l.text = str(my_l) 
        
        for frame_num, time_ms in enumerate(timestamps):
            #print(frame_num, time_ms)
            if (bExtractFrame or bSaveAnno):
                capture.set(0, time_ms)
                ret, frame = capture.read()
                
                ###Zero pad
                if bZeroPad:
                    frame_name = os.path.join(img_dir, seg, str(frame_num).zfill(5)+img_extension)
                else:
                    frame_name = os.path.join(img_dir, seg, str(frame_num)+img_extension)

                cv2.imwrite(frame_name, frame)

            if (bSaveAnno & bTrainSet):
                if (frame_num in labeled_indexes):
                    row_data = data[np.array(labeled_indexes) == frame_num]
                    row_data.reset_index(drop=True, inplace=True)
                    xmin = int(row_data["xmin"].values[0] * vid_w)
                    xmax = int(row_data["xmax"].values[0] * vid_w)
                    ymin = int(row_data["ymin"].values[0] * vid_h)
                    ymax = int(row_data["ymax"].values[0] * vid_h)
                    my_l = 1  
                    x_otb = xmin + 1
                    y_otb = ymin + 1
                    w_otb = xmax - xmin 
                    h_otb = ymax - ymin    

                    if (bDrawBB):
                       cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                       cv2.imwrite(frame_name.replace('imgs', 'imgs_bb'), frame)
                           
                    # XML bounding box polygon
                    polygon = ET.SubElement(obj, 'polygon') 
                    t = ET.SubElement(polygon, 't') 
                    t.text = str(frame_num)
                    pt = ET.SubElement(polygon, 'pt') 
                    x = ET.SubElement(pt, 'x') 
                    y = ET.SubElement(pt, 'y') 
                    l = ET.SubElement(pt, 'l') 
                    x.text = str(xmin)
                    y.text = str(ymin)
                    l.text = str(my_l)
                    pt = ET.SubElement(polygon, 'pt') 
                    x = ET.SubElement(pt, 'x') 
                    y = ET.SubElement(pt, 'y') 
                    l = ET.SubElement(pt, 'l') 
                    x.text = str(xmin)
                    y.text = str(ymax)
                    l.text = str(my_l) 
                    pt = ET.SubElement(polygon, 'pt') 
                    x = ET.SubElement(pt, 'x') 
                    y = ET.SubElement(pt, 'y') 
                    l = ET.SubElement(pt, 'l') 
                    x.text = str(xmax)
                    y.text = str(ymax)
                    l.text = str(my_l) 
                    pt = ET.SubElement(polygon, 'pt') 
                    x = ET.SubElement(pt, 'x') 
                    y = ET.SubElement(pt, 'y') 
                    l = ET.SubElement(pt, 'l') 
                    x.text = str(xmax)
                    y.text = str(ymin)
                    l.text = str(my_l)              


                else:
                    my_l, xmin, xmax, ymin, ymax = 0, 0, 0, 0, 0
                    x_otb, y_otb, w_otb, h_otb = 0, 0, 0, 0


                otb_rows += [ [ str(int(x_otb)), str(int(y_otb)), str(int(w_otb)), str(int(h_otb)) ] ]

            # create a new XML file with the results

        capture.release()

        if (bSaveAnno):          
            xml_data = ET.tostring(annotation, encoding="unicode") #unicode encoding != 'UTF-8':
            with open(xml_path, "w")  as myfile:
                myfile.write(xml_data) 

            df = pd.DataFrame(otb_rows)
            df.to_csv(otb_path, index=False, header=False)
          

        if (bZipFrame):
            frame_names = [f for f in os.listdir(os.path.join(img_dir, seg)) if f.endswith('.jpg')]
            if (os.path.exists(zip_path)): # Check if exists
                with zipfile.ZipFile(zip_path, 'r') as myzip:
                    nb_frames = len(frame_names)
                    nb_files = len(myzip.infolist())
                    if (nb_frames == nb_files): 
                        bZipFrame = False # Already exists with same number of frames! => considered as complete
                        if (bCheckFiles):
                            if (has_zero_file(frame_path)):
                                bZipFrame = True
                                print("Replacing zip due to bad frames:", seg)           
                    else:
                        print("Replacing zip due to missing frames:", seg)
                    '''
                    ret = myzip.testzip()
                    if ret is not None:
                        print("Replacing corrupted zip file:", seg)
                        bZipFrame = True         
                    '''
        
        if (bZipFrame):
            with zipfile.ZipFile(zip_path, 'w') as myzip:
                for f in frame_names:   
                    myzip.write(filename=os.path.join(img_dir, seg, f), arcname=f)


    # Remove the temporary video
    if bRemove:
        os.remove(video_path)
        

    return vid


# Parse the annotation csv file and schedule downloads and cuts
def main(input_csv, output_folder, num_threads=4, bRemove = False, bSaveAnno = False, bExtractFrame = False, bZipFrame = False, bZeroPad = True, bCheckFiles = False):
    """Download the entire youtube-bb data set into `output_folder`.
    """

    imgs_folder = os.path.join(output_folder, "imgs")
    anno_folder = os.path.join(output_folder, "anno")
    vids_folder = os.path.join(output_folder, "vids")
    zips_folder = os.path.join(output_folder, "zips")

    os.makedirs(vids_folder, exist_ok=True) 
    if (bExtractFrame): 
        os.makedirs(imgs_folder, exist_ok=True)  
    if (bZipFrame): 
        os.makedirs(zips_folder, exist_ok=True)    
    if (bSaveAnno): 
        os.makedirs(anno_folder, exist_ok=True)
        os.makedirs(anno_folder.replace('anno', 'anno_otb'), exist_ok=True)



    df = pd.DataFrame.from_csv(input_csv, header=None, index_col=False)
    if len(df.columns) == 6: 
        df.columns = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence']
        bTrainSet = False
        bDrawBB = False
        
    elif len(df.columns) == 10: 
        df.columns = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence', "xmin" , "xmax" , "ymin" , "ymax" ]
        bTrainSet = True
        bDrawBB = True
        
    else:
        raise Exception("Number of column in csv file not 6 not 10")
             
    bPrintProgress = False
    
    # Get list of unique video files
    df['segment_id'] = df['youtube_id'].map(str) + "_" + df['object_id'].map(str)
    vids = df['youtube_id'].unique()
    segs = df['segment_id'].unique()
    
    print('Number of videos:',len(vids))
    print('Number of segments:',len(segs))

    if (num_threads == 1):
        [dl_and_cut(df[df['segment_id']==seg], output_folder, imgs_folder, anno_folder, vids_folder, zips_folder, bRemove, bSaveAnno, bExtractFrame, bZipFrame, bZeroPad, bCheckFiles, bTrainSet, bDrawBB) for seg in tqdm(segs)]

    else:
        print('Downloading videos in parallel threads')
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(dl_and_cut, df[df['youtube_id']==vid], output_folder, imgs_folder, anno_folder, vids_folder, zips_folder) for vid in vids]

            with tqdm(total=len(vids)) as pbar:
                for i, _ in tqdm(enumerate(futures.as_completed(fs))):
                    pbar.update()
            
            if (bPrintProgress):
                for i, f in enumerate(futures.as_completed(fs)):
                    # Write progress to error so that it can be seen
                    printProgress(i, len(vids),
                                prefix = input_csv,
                                suffix = 'Done\n',
                                barLength = 30)
                            
        print('Extracting frames in parallel threads')
        print('Remove videos after frame extraction:', bRemove)
        print('Save annotations:', bSaveAnno)
        print('Extract video frames:', bExtractFrame)
        print('Zip video frames:', bZipFrame)
        print('Zeropadding for filenames:', bZeroPad)
        print('Check for corrupt frames (slow):', bCheckFiles)
        print('Training set (True) / Test set (False):', bTrainSet)
        print('Visualize annotated bounding boxes:', bDrawBB)
        
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(dl_and_cut, df[df['segment_id']==seg], output_folder, imgs_folder, anno_folder, vids_folder, zips_folder, bRemove, bSaveAnno, bExtractFrame, bZipFrame, bZeroPad, bCheckFiles, bTrainSet, bDrawBB) for seg in segs]
            with tqdm(total=len(segs)) as pbar:
                for i, _ in tqdm(enumerate(futures.as_completed(fs))):
                    pbar.update()
            
            if (bPrintProgress):
                for i, f in enumerate(futures.as_completed(fs)):
                    # Write progress to error so that it can be seen
                    printProgress(i, len(segs),
                                prefix = input_csv,
                                suffix = 'Done\n',
                                barLength = 30)

    print( output_folder + ': All videos downloaded')



if __name__ == '__main__':
    # python DownloadVideosTraining.py Dataset_Train\Dataset_Train_0.csv Dataset_Train\Train_0 --save_annotation --extract_frame --zip_frame --num_threads=10
    p = argparse.ArgumentParser(description='Download Youtube Videos\n Exemple: python DownloadVideosTraining.py Dataset_Train\Dataset_Train_0.csv Dataset_Train\Train_0 --save_annotation --extract_frame --zip_frame --num_threads=10')
    p.add_argument('input_csv', type=str, 
        help='CSV File in YT-BB format containing the list of videos.')
    p.add_argument('output_folder', type=str,
        help='Folder where features will be saved.')
    p.add_argument('--num_threads', required=False, type=int, default=1,
        help='Number of parallel threads.')

    p.add_argument('--remove_video', required=False, action='store_true',
        help='Remove the video after extracting the frames' )
    p.add_argument('--save_annotation', required=False, action='store_true',
        help='Save the annotation too' )
    p.add_argument('--extract_frame', required=False, action='store_true',
        help='Extract the frame from the video' )
    p.add_argument('--zip_frame', required=False, action='store_true',
        help='Extract the frame from the video' )
    args = p.parse_args()


    # print(args)
    main(input_csv = args.input_csv, 
        output_folder = args.output_folder, 
        num_threads = args.num_threads, 
        bRemove = args.remove_video, 
        bSaveAnno = args.save_annotation, 
        bExtractFrame = args.extract_frame, 
        bZipFrame = args.zip_frame)

