#from jetson_inference import detectNet
#from jetson_utils impor videoSource,videoOutput
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.7)

image_path = "/home/nvidia/jetson-inference/data/images/room_4.jpg"
img = jetson.utils.loadImage(image_path)


    
detections = net.Detect(img)
print(detections)

output_path = "/home/nvidia/jetson-inference/examples/room_4_detected_0.7.jpg"
jetson.utils.saveImage(output_path, img)
