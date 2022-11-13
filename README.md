# collatzviewer

20220913 version 1.0 released

Requirements:

CUDA GPU recommended

install pytorch with cuda: https://pytorch.org/features/

ie: pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117



arrowkeys: pan/scroll image

left mouse button: increment collatz n

right mouse button: decrement collatz n

mousewheel scroll: zoom image in/out




future improvements:

add OneAPI or other GPU libraries

replace "hsv_to_rgb255" function with a lookup table and/or torch code

replace pygame to use higher resolution image display

try 128bit floats on GPU for deeper image zooming

improve float overflow detection

check collatz values for black pixels to add colour or transparency etc

render higher resolution images when not pan/scroll/zooming

add button to save image with optional text overlay

add gui to input collatz n value, x y and zoom, resolution etc

todo:

add screenshot


record and upload to youtube, link to video
