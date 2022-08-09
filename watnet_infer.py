## author: xin luo, creat: 2021.8.11

'''
des: perform surface water mapping by using pretrained watnet
     through funtional api and command line, respectively.

example:
     funtional api:
        water_map = watnet_infer(rsimg) 
     command line: 
        python watnet_infer.py data/test-demo/*.tif
        python watnet_infer.py data/test-demo/*.tif -o data/test-demo/result
    note: 
        rsimg is np.array (row,col,band), value: [0,1]
        data/test-demo/*.tif is the sentinel-2 image path
        data/test-demo/result is output directory
'''

import os
import numpy as np
import tensorflow as tf
import argparse
from utils.imgPatch import imgPatch
from utils.geotif_io import readTiff,writeTiff
import matplotlib.pyplot as plt
import rioxarray
import pyproj
import rasterio
import streamlit as st
from io import BytesIO
import pandas as pd

import pydaisi as pyd
sentinel_s2_l2a_cogs = pyd.Daisi("laiglejm/Sentinel S2 l2a COGS")

## default path of the pretrained watnet model
path_watnet = 'model/pretrained/watnet.h5'

def get_args():

    description = 'surface water mapping by using pretrained watnet'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        'ifile', metavar='ifile', type=str, nargs='+',
        help=('file(s) to process (.tiff)'))

    parser.add_argument(
        '-m', metavar='watnet', dest='watnet', type=str, 
        nargs='+', default=path_watnet, 
        help=('pretrained watnet model (tensorflow2, .h5)'))

    parser.add_argument(
        '-o', metavar='odir', dest='odir', type=str, nargs='+', 
        help=('directory to write'))

    return parser.parse_args()


def watnet_infer(rsimg, path_model = path_watnet):

    ''' des: surface water mapping by using pretrained watnet
        arg:
            img: np.array, surface reflectance data (!!data value: 0-1), 
                 consist of 6 bands (blue,green,red,nir,swir-1,swir-2).
            path_model: str, the path of the pretrained model.
        retrun:
            water_map: np.array.
    '''
    ###  ----- load the pretrained model -----#
    model = tf.keras.models.load_model(path_model, compile=False)
    ### ------ apply the pre-trained model
    imgPatch_ins = imgPatch(rsimg, patch_size=512, edge_overlay=80)
    patch_list, start_list, img_patch_row, img_patch_col = imgPatch_ins.toPatch()
    result_patch_list = [model(patch[np.newaxis, :]) for patch in patch_list]
    result_patch_list = [np.squeeze(patch, axis = 0) for patch in result_patch_list]
    pro_map = imgPatch_ins.toImage(result_patch_list, img_patch_row, img_patch_col)
    water_map = np.where(pro_map>0.5, 1, 0)

    return water_map

def inference(sen2_img):
    
    sen2_img = np.float32(np.clip(sen2_img/10000, a_min=0, a_max=1))  ## normalization
    ## surface water mapping by using watnet
    water_map = watnet_infer(rsimg=sen2_img)

    return water_map

def render_images(item, lat, lon, radius=10000, attribute_to_display = "visual"):
    proj = pyproj.Transformer.from_crs(4326, item.properties['proj:epsg'], always_xy=True)

    x1, y1 = (lon, lat)
    x2, y2 = proj.transform(x1, y1)
    assets = item.assets
    visual_href = assets[attribute_to_display].href
    visual = rioxarray.open_rasterio(visual_href)

    visual_clip = visual.rio.clip_box(minx=x2-radius,miny=y2-radius,maxx=x2+radius,maxy=y2+radius)
    if attribute_to_display == 'visual':
        visual_clip.plot.imshow(figsize=(10,10), cmap = 'viridis')
        plt.show()
    return visual_clip
    # print(item.datetime)
    
    # else:
    #     # print(visual_clip['band'])
    #     visual_clipp = visual_clip.where(visual_clip.data == 6)
    #     print(np.nansum(visual_clipp.data))
    # #     visual_clipp[0].plot.imshow(figsize=(10,10), cmap = 'viridis')
    # plt.show()


def prepare_data(item, lat, lon, radius=10000, attribute_to_display = "visual"):
    proj = pyproj.Transformer.from_crs(4326, item.properties['proj:epsg'], always_xy=True)

    x1, y1 = (lon, lat)
    x2, y2 = proj.transform(x1, y1)
    assets = item.assets

    data_array = []
    for a in ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']:
            st.write(f"Collecting {a}")
            visual_href = assets[a].href
            visual = rioxarray.open_rasterio(visual_href)
            visual_clip = visual.rio.clip_box(
                            minx=x2-radius,
                            miny=y2-radius,
                            maxx=x2+radius,
                            maxy=y2+radius
                        )
            data = visual_clip.data[0]
            if a == 'B02':
                target_shape = data.shape
            if a == 'B11' or a == 'B12':
                    data = np.repeat(data, 2, axis = 1)
                    data = np.repeat(data, 2, axis = 0)
                    cut_row = data.shape[0] - target_shape[0]
                    cut_col = data.shape[1] - target_shape[1]
                    data = data[:-cut_row,:-cut_col]
            data_array.append(data)
    data_array = np.array(data_array)
    data_array = data_array.transpose().reshape((data_array.shape[1], data_array.shape[2], data_array.shape[0]))

    return data_array


def st_ui():
    st.title("Surface water analysis through time")

    data_dict={"McConaughy Lake, NB":[41.2394, -101.7777],
                 "Los Vaqueros reservoir, CA":[37.8190, -121.7348],
                 "Lake Nacimiento, CA":[35.7477, -120.9296]}

    lake_select = st.sidebar.selectbox("Select a reservoir", list(data_dict.keys()))

    lat, lon = data_dict[lake_select]
    radius = 5000
    data_points = []
    time_points = []
    for year in [2018, 2019, 2020, 2021, 2022]:
        for month in ['05', '10']:
            if month == '05':
                datetime=f"{year}-05-01/{year}-07-30"
            else:
                datetime=f"{year}-10-01/{year}-12-30"
            print(datetime)
            # try:
            item = sentinel_s2_l2a_cogs.get_sat_images(datetime, lat, lon).value
            st.write("Done collecting")
            
            data_array = prepare_data(item, lat, lon, radius)
            st.write("Done preparing")
            water_map = inference(data_array)
            st.write("Done infering")
            visual_clip = render_images(item, lat, lon, radius, attribute_to_display = "visual")
            st.write("Done rendering")
            data_points.append(np.nansum(water_map))
            time_points.append(item.datetime)
            st.write(f"On {item.datetime}, relative area is : {np.nansum(water_map)}")
            visual_clip = visual_clip.data.transpose().reshape(visual_clip.shape[1], visual_clip.shape[2], visual_clip.shape[0])
            fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
            ax1.imshow(np.rot90(visual_clip, k=3))
            ax2.imshow(np.rot90(water_map, k=3))
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', transparent = True)
            st.image(buf, use_column_width=False)

            #     plt.show()

            # except Exception as e:
            #     print(e)
            #     print(f"Nothing found for year {year} and month {month}")
            #     continue

    # chart_data = pd.DataFrame(
    # data_points,
    # columns=time_points)

    # st.bar_chart(chart_data)




if __name__ == '__main__':

    st_ui()


    

    # lat, lon = 41.2394, -101.7777 # McConaughy Lake, NB, +++
    # lat, lon = 37.8190, -121.7348 # Los Vaqueros reservoir, CA +++
    # lat, lon = 35.7477, -120.9296 # Lake Nacimiento, CA # +++
    # radius = 5000

    # for year in [2018, 2019, 2020, 2021, 2022]:
    #     for month in ['05', '10']:
    #         if month == '05':
    #             datetime=f"{year}-05-01/{year}-07-30"
    #         else:
    #             datetime=f"{year}-10-01/{year}-12-30"
    #         print(datetime)
    #         try:
    #             item = sentinel_s2_l2a_cogs.get_sat_images(datetime, lat, lon).value
    #             print("Done collecting")
    #             data_array = prepare_data(item, lat, lon, radius)
    #             print("Done preparing")
    #             water_map = inference(data_array)
    #             print("Done infering")
    #             visual_clip = render_images(item, lat, lon, radius, attribute_to_display = "visual")
    #             print("Done rendering")
    #             print(np.nansum(water_map))
    #             visual_clip = visual_clip.data.transpose().reshape(visual_clip.shape[1], visual_clip.shape[2], visual_clip.shape[0])
    #             fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
    #             ax1.imshow(np.rot90(visual_clip, k=3))
    #             ax2.imshow(np.rot90(water_map, k=3))

    #             plt.show()

    #         except Exception as e:
    #             print(e)
    #             print(f"Nothing found for year {year} and month {month}")
    #             continue

    # args = get_args()
    # ifile = args.ifile
    # path_model = args.watnet
    # odir = args.odir
    # ## write path
    # if odir:
    #     if not os.path.exists(odir[0]):
    #         os.makedirs(odir[0])
    #     ofile = [os.path.splitext(file)[0] + '_water.tif' for file in ifile]
    #     ofile = [os.path.join(odir[0], os.path.split(file)[1]) for file in ofile]
    # else:
    #     ofile = [os.path.splitext(file)[0] + '_water.tif' for file in ifile]

    # for i in range(len(ifile)):
    #     print('file in -->', ifile[i])
        
    #     ## image reading and normalization
    #     sen2_img, img_info = readTiff(path_in=ifile[i])
    #     print(sen2_img.shape)
    #     sen2_img = np.load('../test_array.npy')
    #     sen2_img = np.float32(np.clip(sen2_img/10000, a_min=0, a_max=1))  ## normalization
    #     ## surface water mapping by using watnet
    #     water_map = watnet_infer(rsimg=sen2_img)
    #     # write out the result
    #     print('write out -->', ofile[i])
    #     print(100*np.nansum(water_map)/1e6, "surface in km2")
    #     plt.imshow(water_map)
    #     plt.show()
    #     # writeTiff(im_data = water_map.astype(np.int8), 
    #     #         im_geotrans = img_info['geotrans'], 
    #     #         im_geosrs = img_info['geosrs'], 
    #     #         path_out = ofile[i])


