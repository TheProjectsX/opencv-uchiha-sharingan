# Sharingan of the Uchiha!

As a `Naruto` Fan, I wanted to do **Naruto** projects using OpenCV..
Guess what? The first project Created is: Sharingan effect in your Eyes!

### Technologies used:

- OpenCV
- Mediapipe

### Packages used:

- opencv-python
- mediapipe
- cvzone
- numpy

```shell script
pip install opencv-python mediapipe cvzone numpy
```

### Features:

Added 8 Sharingan Images to use, with and without Iris.

Every Blink will change the Sharingan Version.

Eyes are sorted from level 1 to level 99

### Running the program

After Cloning / Downloading the repo, open cmd in the folder and run:

```shell script
python main.py --source <Frame-Source> --source-path <Frame-Source-Path>
```

- `<Frame-Source>` has 2 options: `image` and `video`
- `<Frame-Source-Path>` can be:
  - An Image File Path (for --source image)
  - A Video File Path (for --source video)
  - Camera Index (for --source video)

### Example:

```shell script
python main.py --source video --source-path 1
```

But if ran without any arguments, video will be the source and 0 will be the camera index

```shell script
python main.py
```

### Message:

More customizations can be performed to make it accurate
