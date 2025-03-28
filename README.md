# SelfDrivingSim 🚗✨

Buckle up and welcome to **SelfDrivingSim**—where AI meets the open road! This isn’t just a project; it’s a thrilling ride into the world of autonomous driving. We’re training a convolutional neural network (CNN) to predict steering angles from raw images, turning pixels into precision. Built with Python, TensorFlow, and a sprinkle of magic, this is your ticket to exploring the future of self-driving tech!

![self-driving-sim Banner](https://github.com/srivatsa2512/self-driving-sim)  
*Rev up your engines! Placeholder: Swap this with a slick banner of a car cruising or a neural net in action.*

## 🌟 Why This Rocks

Picture this: a car that *learns* to steer itself, dodging curves and hugging lanes—all from a pile of images and some clever code. **SelfDrivingSim** takes driving data, jazzes it up with augmentation, and feeds it to a CNN that’s ready to roll. Inspired by the pros at NVIDIA, this project is for dreamers, coders, and anyone who’s ever yelled, “Why can’t my car drive itself?!”

### What Makes It Awesome
- **Data Wizardry:** Balances steering angles so our AI doesn’t just go straight forever.
- **Augmentation Fun:** Zooms, pans, and flips images like a Hollywood director.
- **Brain Power:** A CNN that predicts steering like a seasoned driver.
- **Simulator Vibes:** Hooks up with tools like Udacity’s driving sim—vroom vroom!

## 🛠️ Jump In—Let’s Build It!

Ready to take the driver’s seat (figuratively, of course)? Here’s how to get **SelfDrivingSim** roaring on your setup.

### What You’ll Need
- **Python 3.8+**: The engine under the hood.
- **Libraries**: A pit crew of tools—install them with `pip`.
- **Dataset**: Grab some driving data (like `driving_log.csv` and an `IMG` folder) to fuel the ride.

### Setup—Zero to Hero
1. **Snag the Code:**
   ```bash
   git clone https://github.com/srivatsa2512/self-driving-sim.git
   cd SelfDrivingSim

2.**Gear Up a Virtual Environment (Optional but Cool):**
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3.**Load the Toolbox:**
pip install -r requirements.txt
tensorflow
numpy==1.26.4  # Keeps imgaug happy
pandas
matplotlib
scikit-learn
opencv-python
imgaug

4.**Drop in Your Data:**
Toss your myData folder (with driving_log.csv and IMG) into the project root, or tweak path in trainingSimulation.py to point where your data’s parked.
.
🚀 Hit the Gas—Run It!
Time to see this baby in action!

### **Fire It Up:**
```bash
python trainingSimulation.py


Setting up...
Total images Imported: 3132
Removed Images: 1328
Remaining images: 1804
Total training images: 1443
Total validation images: 361
<Model Summary—Layers Galore!>
Epoch 1/2
20/20 [==============================] - Xs - loss: Y.YYYY - val_loss: Z.ZZZZ


