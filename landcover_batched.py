import numpy as np
from osgeo import gdal, gdal_array, osr, ogr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import joblib
import os

# Tell GDAL to throw Python exceptions and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


def predict_save(input_tif):
    # Function to process and predict land cover using trained rf model
    # Output: classified raster
    # Warning: function creates files and directories in directory of input image -
    #   none of which are deleted automatically


    # Open image to be classified and read as array
    img2_file = os.path.basename(input_tif)     # Extracting file name
    local_path = os.path.dirname(input_tif)     # Directory containing raster file
    img2_raster = gdal.Open(input_tif, gdal.GA_ReadOnly)    # Opening raster file in gdal

    # Initializing numpy array of raster dimensions
    img2 = np.zeros((img2_raster.RasterYSize, img2_raster.RasterXSize, img2_raster.RasterCount),
                    gdal_array.GDALTypeCodeToNumericTypeCode(img2_raster.GetRasterBand(1).DataType))
    for b in range(img2.shape[2]):  # Reading raster values into numpy array
        img2[:, :, b] = img2_raster.GetRasterBand(b + 1).ReadAsArray()

    # Reshape raster into 2d array (nrow * ncol, nband) for classification
    new_shape = (img2.shape[0] * img2.shape[1], img2.shape[2])
    img2_as_array = img2[:, :, :3].reshape(new_shape)

    # Splitting raster array into batches for classification, for memory efficiency
    divs = 100  # Number of files to split array into
    split = int(img2_as_array.shape[0] / divs)

    # Creating folder for processed files in directory of input raster
    dir_path = os.path.join(local_path, 'batched_' + img2_file)
    os.mkdir(dir_path)

    # Iterating through batches of raster array and saving to local csv file
    iter = 0
    for batch in range(divs):
        batch_file = os.path.join(dir_path, 'batch_' + str(batch) + '.csv')
        img2_batch = img2_as_array[(batch * split): ((batch + 1) * split)]  # Splitting array in batches
        batch_prediction = rf.predict(img2_batch)   # Classifying batch using rf model
        np.savetxt(batch_file, batch_prediction)    # Saving batch predictions to csv file
        iter = batch + 1

    # Processing end of array (if not already included)
    excess_batch = img2_as_array[(iter * split):]  # Splitting array in batches
    batch_prediction = rf.predict(excess_batch)   # Classifying batch using rf model
    excess_file = os.path.join(dir_path, 'batch_' + str(iter) + '.csv')
    np.savetxt(excess_file, batch_prediction)    # Saving batch predictions to csv file

    # Reading back into one complete array from saved csv files
    img2_classed = np.empty(0, dtype=np.uint8)  # Creating numpy array to append prediction values to
    for file in range(divs):
        read_file = os.path.join(dir_path, 'batch_' + str(file) + '.csv')
        read_arr = np.loadtxt(read_file)
        arr_length = len(read_arr)
        img2_classed = np.append(img2_classed, read_arr)    # Appending csv values to full array
        print('File {file}: \t array type = {arr_type} \t file length = {arr_length} \t total length = {full_len}'.format(
                arr_type=type(read_arr), file=file, arr_length=arr_length, full_len=len(img2_classed)))

    # Picking up last csv file omitted from loop
    if os.path.isfile(excess_file):
        read_excess = np.loadtxt(excess_file)
        img2_classed = np.append(img2_classed, read_excess)
        print('File {file}: \t array type = {arr_type} \t file length = {arr_length} \t total length = {full_len}'.format(
                arr_type=type(read_excess), file=file, arr_length=len(read_excess), full_len=len(img2_classed)))

    img2_classed = img2_classed.astype(np.uint8)    # Ensuring all prediction values read from csv are type int8

    # Reshaping full array to initial raster size
    img2_classed = img2_classed.reshape(img2[:, :, 0].shape)

    # Converting array to raster and saving
    classified_folder = os.path.join(local_path, 'classified')  # Directory for classified images
    if not os.path.exists(classified_folder):  # Creating folder for classified images if not already created
        os.mkdir(classified_folder)

    classified = os.path.join(classified_folder, 'landcover_' + img2_file + '.tif')     # Path of classified image
    driver = gdal.GetDriverByName("GTiff")
    metadata = driver.GetMetadata()

    dst_ds = driver.Create(classified, xsize=img2_raster.RasterXSize, ysize=img2_raster.RasterYSize,
                           bands=1, eType=gdal.GDT_Byte)    # Creating raster file

    # Setting raster spatial/geotransform data
    dst_ds.SetGeoTransform(img2_raster.GetGeoTransform())
    srs = osr.SpatialReference()
    srs.SetUTM(15, 1)   # TODO Change based on input raster UTM
    srs.SetWellKnownGeogCS("WGS84")   # TODO Change based on input raster CRS
    dst_ds.SetProjection(img2_raster.GetProjection())
    # raster = np.zeros((15102, 73788), dtype=np.uint8)
    dst_ds.GetRasterBand(1).WriteArray(img2_classed)
    dst_ds = None   # Close dataset


# TRAINING RANDOM FOREST MODEL

# TODO Change to your image and training file paths
#img_raster = gdal.Open('/Volumes/urbansky/USKY01_D00095_010_13T_20230822_20230909125830_A0_000_000_01.tiff', gdal.GA_ReadOnly)
img_raster = gdal.Open('/Volumes/URBAN_SKY/URBANSKY3/laporte.tif', gdal.GA_ReadOnly)
#train_raster = gdal.Open('/Volumes/URBAN_SKY/poudre_image/full_training_raster.tif', gdal.GA_ReadOnly)
train_raster = gdal.Open('/Volumes/URBAN_SKY/URBANSKY3/python/training_poly_raster.tif', gdal.GA_ReadOnly)

# Read bands into array
img = np.zeros((img_raster.RasterYSize, img_raster.RasterXSize, img_raster.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_raster.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_raster.GetRasterBand(b + 1).ReadAsArray()

train_roi = train_raster.GetRasterBand(1).ReadAsArray()

# Clip arrays to training polygons
x_raw = img[train_roi > 0, :]
img = 0     # Delete image array after clipping to free memory
y_raw = train_roi[train_roi > 0]
train_roi = 0     # Delete image array after clipping to free memory

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(x_raw, y_raw, shuffle=True, train_size=0.75)

# Initialize rf model
rf = RandomForestClassifier(n_estimators=600,
                            oob_score=True,
                            max_depth=20,
                            random_state = 0,
                            verbose = 0)

# Fit rf model
model = rf.fit(X_train, Y_train)

predict_save('/Volumes/urbansky/poudre_5_23/URBANSKY_USKY01_MS_20230505T1743_EuvrtK/URBANSKY_USKY01_MS_20230505T1743_Tile_0_0_b2ba.tif')