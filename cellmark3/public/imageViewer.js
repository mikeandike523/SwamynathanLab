// General guidance for using canvases in react: https://medium.com/@pdx.lucasm/canvas-with-react-js-32e133c05258

MIN_ZOOM = 0.125
MAX_ZOOM = 10.0
ZOOM_RATE = 0.125


function ImageViewer(props) {

    const canvasRef = useRef(null)

    const forceRefreshSwitch = props.forceRefreshSwitch

    const [zoom, setZoom] = [props.zoom, props.setZoom]
    
    const [origin, setOrigin] = [props.origin, props.setOrigin]

    const image = props.image // no need to pass setImage to the ImageViewer\

    const nullPosition = [
        Math.round(props.image.w/2),
        Math.round(props.image.h/2)
    ]

    const [initMousePosition, setInitMousePosition] = useState(nullPosition)
    const [initOrigin, setInitOrigin] = useState(nullPosition)
    const [dragging, setDragging] = useState(false)

    const getCoordRelativeCanvas=(clientX, clientY)=>{
        const boundingRect =canvasRef.current.getBoundingClientRect()
        return [clientX-boundingRect.left, clientY-boundingRect.top]
    }

    const preciseMapCanvasCoordinateToImageCoordinateAtZoom = (canvasX, canvasY, zoomLevel)=>{
        const offsetX = canvasX-image.w/2
        const offsetY = canvasY-image.h/2
        const scaledOffsetX = offsetX/zoomLevel
        const scaledOffsetY = offsetY/zoomLevel
        return [
       origin[0]+scaledOffsetX,
     origin[1]+scaledOffsetY
        ]    
    }

    const preciseMapImageCoordinateToCanvasCoordinateAtZoom = (imageX, imageY, zoomLevel)=>{
        const diffX = imageX - origin[0]
        const diffY = imageY - origin[1]
        const screenDiffX = diffX * zoomLevel
        const screenDiffY = diffY * zoomLevel
        return [
         image.w/2+screenDiffX,
           image.h/2+screenDiffY
        ]
    }

    const handleCanvasMouseDown = (evt) => {
        if(evt.button===2){
            setInitMousePosition([evt.clientX, evt.clientY])
            setInitOrigin(origin)
            setDragging(true)
        }
        if(evt.button===0){

            const coord = getCoordRelativeCanvas(evt.clientX, evt.clientY)

            var imageCoord = preciseMapCanvasCoordinateToImageCoordinateAtZoom(coord[0],coord[1],zoom)

            if(!evt.shiftKey){
                props.markAtCoord(imageCoord[0],imageCoord[1])
            }else{
                props.eraseAtCoord(imageCoord[0],imageCoord[1])
            }
        }
    }

    const handleCanvasMouseMove = (evt) => {
        if(dragging){
            var screenDiffX = evt.clientX - initMousePosition[0]
            var screenDiffY = evt.clientY - initMousePosition[1]
            var diffX = screenDiffX/zoom
            var diffY = screenDiffY/zoom
            var newOrigin = [
                initOrigin[0]-diffX,
                initOrigin[1]-diffY
            ]
            setOrigin(newOrigin)
        }
    }

    const performFixedPointZoom = (fixedPointX, fixedPointY, oldZoom, newZoom) => {
        const [mappedFixedPointX, mappedFixedPointY] =  preciseMapCanvasCoordinateToImageCoordinateAtZoom(fixedPointX,fixedPointY,oldZoom)
        const [newUnmappedFixedPointX, newUnmappedFixedPointY] =  preciseMapImageCoordinateToCanvasCoordinateAtZoom(mappedFixedPointX,mappedFixedPointY,newZoom)

        const screenSpaceDeltas = [
            newUnmappedFixedPointX - fixedPointX,
            newUnmappedFixedPointY - fixedPointY
        ]

        
        const newOrigin = [
            origin[0] + screenSpaceDeltas[0]/newZoom,
            origin[1] + screenSpaceDeltas[1]/newZoom
        ]

        setOrigin(newOrigin)
        setZoom(newZoom)

    }

    const handleCanvasMouseWheel = (evt) => {
        const deltaY = evt.deltaY
        const delta = -signum(deltaY)*ZOOM_RATE;
         
        let oldZoom = zoom;
        let newZoom = clipValue(oldZoom+delta, MIN_ZOOM, MAX_ZOOM)

        const [cX, cY] = getCoordRelativeCanvas(evt.clientX,evt.clientY)

        performFixedPointZoom(cX, cY, oldZoom, newZoom)

    }

    const handleCanvasMouseUp = (evt) => {
        if(evt.button===2){
            setDragging(false)
        }
    }

    const draw=ctx=>{

        // image and cavas size will be synced due to how react works

        // Start by getting html5 imageData object
        const imageData = ctx.getImageData(0, 0, image.w, image.h)

        // Define a function to write an rgb value to the imageData
        const setImageDataPixel = (x, y, r, g, b)=>{
            imageData.data[y * image.w * 4 + x * 4 + 0] = r
            imageData.data[y * image.w * 4 + x * 4 + 1] = g
            imageData.data[y * image.w * 4 + x * 4 + 2] = b
            imageData.data[y * image.w * 4 + x * 4 + 3] = 255
        }

        // loop x and y over 0 to image.w and 0 to image.h
        for (let x = 0; x < image.w; x++) {
            for (let y = 0; y < image.h; y++) {
                var [imageX, imageY] = preciseMapCanvasCoordinateToImageCoordinateAtZoom(x,y,zoom)
                imageX = Math.round(imageX)
                imageY = Math.round(imageY)
                const color = image.getPixel(imageX,imageY)
                setImageDataPixel(x, y, color[0], color[1], color[2])
            }
        }

        // Blit the imageData using ctx and the putImageData function
        ctx.putImageData(imageData, 0, 0)

    }

    useEffect(()=>{

        const canvas = canvasRef.current

        if(canvas){ // Is this needed?

            const ctx = canvas.getContext('2d',{
                willReadFrequently:true
            })

            draw(ctx)
        }
    },[draw,zoom,origin,image,forceRefreshSwitch])


    return props.visible?R.cE('canvas', {
        ref: canvasRef,
        style: {
            width: image.w + "px",
            height: image.h + "px",
            border: "2px dotted black"
        },
        width: image.w + "",
        height: image.h + "",
        onMouseDown:handleCanvasMouseDown,
        onMouseMove:handleCanvasMouseMove,
        onMouseUp:handleCanvasMouseUp,
        onWheel:handleCanvasMouseWheel,
        onContextMenu:(evt)=>{
            evt.preventDefault()
        }
    }):Fragment()()

}

function newImageViewer(imageViewerStateObj,
        markAtCoord,
        eraseAtCoord,
    ) {
    return R.cE(ImageViewer, {
        image: imageViewerStateObj.image,
        setImage: imageViewerStateObj.setImage,
        zoom: imageViewerStateObj.zoom,
        setZoom: imageViewerStateObj.setZoom,
        origin: imageViewerStateObj.origin,
        setOrigin: imageViewerStateObj.setOrigin,
        forceRefreshSwitch: imageViewerStateObj.forceRefreshSwitch,
        visible: imageViewerStateObj.visible,
        markAtCoord: markAtCoord,
        eraseAtCoord: eraseAtCoord,
    })
}

function useImageViewerState(initialImage = new Image(0, 0)) {
    const [image, setImage] = useState(initialImage)
    const [zoom, setZoom] = useState(1)
    const [origin, setOrigin] = useState([
        Math.round(image.w/2),
        Math.round(image.h/2)
    ])
    const [forceRefreshSwitch,setForceRefreshSwitch] = useState(false)
    const [visible, setVisible] = useState(false)
    return {
        image: image,
        setImage: setImage,
        zoom: zoom,
        setZoom: setZoom,
        origin: origin,
        setOrigin: setOrigin,
        visible: visible,
        setVisible: setVisible,
        show:()=>{setVisible(true)},
        forceRefresh:()=>{
            setForceRefreshSwitch(!forceRefreshSwitch)
        }
    }
}