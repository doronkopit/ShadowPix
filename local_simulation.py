import maya.cmds as cmds
import random
import string

def setLights(lightPosX, lightNegX, lightPosY, lightNegY, mode="+X"):
    cmds.directionalLight(lightPosX, e=True, intensity=1.8 if mode == "+X" else 0)
    cmds.directionalLight(lightNegX, e=True, intensity=1.8 if mode == "-X" else 0)
    cmds.directionalLight(lightPosY, e=True, intensity=1.8 if mode == "+Y" else 0)
    cmds.directionalLight(lightNegY, e=True, intensity=1.8 if mode == "-Y" else 0)

N = 5
name = random.choice(string.ascii_uppercase) + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N)) + random.choice(string.ascii_uppercase)
print(name)

res = cmds.file("/Users/yalevy/Downloads/global_learning.obj", i=True, type="OBJ", namespace=name, preserveName=True)
print(res)

# Scaling and rotating
cmds.setAttr(name+':Mesh.scaleX', 0.05)
cmds.setAttr(name+':Mesh.scaleY', 0.05)
cmds.setAttr(name+':Mesh.scaleZ', 0.05)
cmds.setAttr(name+':Mesh.rotateZ', -180)

# Bounding box for translation
minX=cmds.getAttr(name+":Mesh.boundingBoxMinX")
maxX=cmds.getAttr(name+":Mesh.boundingBoxMaxX")
minY=cmds.getAttr(name+":Mesh.boundingBoxMinY")
maxY=cmds.getAttr(name+":Mesh.boundingBoxMaxY")
minZ=cmds.getAttr(name+":Mesh.boundingBoxMinZ")
maxZ=cmds.getAttr(name+":Mesh.boundingBoxMaxZ")

# Translation for symmetry
cmds.setAttr(name+':Mesh.translateY',-minY)
cmds.setAttr(name+':Mesh.translateX',-minX/2)

# Take bounding boxes again
minX=cmds.getAttr(name+":Mesh.boundingBoxMinX")
maxX=cmds.getAttr(name+":Mesh.boundingBoxMaxX")
minY=cmds.getAttr(name+":Mesh.boundingBoxMinY")
maxY=cmds.getAttr(name+":Mesh.boundingBoxMaxY")
minZ=cmds.getAttr(name+":Mesh.boundingBoxMinZ")
maxZ=cmds.getAttr(name+":Mesh.boundingBoxMaxZ")

print("minX=" + str(minX))
print("maxX=" + str(maxX))
print("minY=" + str(minY))
print("maxY=" + str(maxY))
print("minZ=" + str(minZ))
print("maxZ=" + str(maxZ))

# Set directional lights for scene
xyOffset = 10
zOffset = maxZ + 10
lightPosX = cmds.directionalLight(position=(maxX + xyOffset, maxY/2, zOffset)) # +X
lightNegX=cmds.directionalLight(position=(minX - xyOffset, maxY/2, zOffset)) # -X
lightPosY=cmds.directionalLight(position=(0, maxY + xyOffset, zOffset)) # +Y
lightNegY=cmds.directionalLight(position=(0, minY - xyOffset, zOffset)) # -Y

cmds.directionalLight(lightPosX, e=True, rotation=(0,60,0))
cmds.directionalLight(lightNegX, e=True, rotation=(0,-60,0))
cmds.directionalLight(lightPosY, e=True, rotation=(-60,0,0))
cmds.directionalLight(lightNegY, e=True, rotation=(60,0,0))

setLights(lightPosX, lightNegX, lightPosY, lightNegY, mode="+X")


