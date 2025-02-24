from pipeline.pipe import FullPipeline

model= 'yolo11n.pt'
pipeline = FullPipeline(LS_ID=194,epochs=5,batch_size=-1)
pipeline.run()