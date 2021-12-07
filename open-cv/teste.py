import cv2 as cv

carregaAlgoritmo = cv.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
image = cv.imread('fotos/imagem3.jpg')

imagemcinza = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
 
faces = carregaAlgoritmo.detectMultiScale(imagemcinza)

print(faces)


for (x,y,l,a) in faces:
    cv.rectangle(image,(x,y),(x + l, y + a),(0,255,0),2)
cv.imshow('faces',image)
cv.waitKey()