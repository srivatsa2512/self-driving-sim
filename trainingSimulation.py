
print('Setting up...')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs, keep ERRORs visible


from utilis import *  # Imports all functions from utilis.py
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Step 1: Load data
    path = r'myData'  # Relative path; assumes myData is in project folder
    # For absolute path, use e.g., r"C:\Users\User\Documents\myData"
    data = importDataInfo(path)

    # Step 2: Balance data
    data = balanceData(data, display=False)

    # Step 3: Load images and steering angles
    imagesPath, steerings = loadData(path, data)

    # Step 4: Split into training and validation sets
    xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
    print('Total training images:', len(xTrain))
    print('Total validation images:', len(xVal))

    # Step 5 & 6: Data augmentation and preprocessing handled in batchGen and preProcessing

    # Step 7: (Optional placeholder for additional logic)

    # Step 8: Create model
    model = createModel()
    model.summary()

    # Step 9: Train model
    history = model.fit(batchGen(xTrain, yTrain, 10, 1),
              steps_per_epoch=300,
              epochs=10,
              validation_data=batchGen(xVal, yVal, 100, 0),
              validation_steps=200)

    # step 10
    model.save('model.h5')
    print('model saved')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training','Validation'])
    plt.ylim([0,1])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()



