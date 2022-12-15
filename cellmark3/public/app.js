// some code copied from https://getbootstrap.com/docs/4.0/components/navbar/

ASYNC_OP_MIN_DELAY_MILLIS = 250

function calculate_marker_element(){

    const MARKER_RADIUS = windowGet("MARKER_RADIUS",0)

    const MARKER_ELEMENT = []

    for(var r = 0; r < 2*MARKER_RADIUS+1;r++){
        MARKER_ELEMENT.push([])
        for(var c = 0; c < 2*MARKER_RADIUS+1;c++){
            let value = ((r-MARKER_RADIUS)*(r-MARKER_RADIUS)+(c-MARKER_RADIUS)*(c-MARKER_RADIUS)) <= (MARKER_RADIUS*MARKER_RADIUS)
            MARKER_ELEMENT[r].push(value)
        }
    }

    windowSet("MARKER_ELEMENT", MARKER_ELEMENT)

}

function App() {

    const [errorMessageComponent, setErrorMessageComponent] = useState(Fragment()())

    const [messageComponent, setMessageComponent] = useState(Fragment()())

    const setErrorMessage = (message) => {
        let components = [
            message?Button({className:"btn btn-link text-secondary",onClick(){
                setErrorMessage("")
            }})("close\u00D7"):Fragment()()
        ]
        let lines = message.split("\n")
        for (let i = 0; i < lines.length; i++) {
            lines[i] = lines[i].replace(/ /g,"\u00A0").replace(/\t/g,"\u00A0\u00A0")
            components.push(Fragment()(lines[i]))
            if(i<lines.length-1){
                components.push(Br()())
            }
        }
        setErrorMessageComponent(Div({className:"text-danger"})(...components))
    }

    const setMessage = (message) => {
        let components = [
            message?Button({className:"btn btn-link text-secondary",onClick(){
                setMessage("")
            }})("close\u00D7"):Fragment()()
        ]
        let lines = message.split("\n")
        for (let i = 0; i < lines.length; i++) {
            lines[i] = lines[i].replace(/ /g,"\u00A0").replace(/\t/g,"\u00A0\u00A0")
            components.push(Fragment()(lines[i]))
            if(i<lines.length-1){
                components.push(Br()())
            }
        }
        setMessageComponent(Div({className:"text-primary"})(...components))
    }


    const [refreshSwitch, setRefreshSwitch] = useState(false)

    const forceRefresh = () => {
        setRefreshSwitch(!refreshSwitch)
    }

    const [image, setImage] = useState(null)
    const [imageFilepath, setImageFilepath] = useState("")
    const [conservativeCellMask, setConservativeCellMask] = useState(null)

    const [initialized, setInitialized] = useState(false)
    const [annotatedImage, setAnnotatedImage] = useState(null)
    const [greyscaleImage, setGreyscaleImage] = useState(null)
    const [marks, setMarks] = useState([])

    const [viewPristineImage, setViewPristineImage] = useState(false)

    const [isSaving, setIsSaving] = useState(false)

    const [isLoading, setIsLoading] = useState(false) 

    const [isAdjusting, setIsAdjusting] = useState(false)

    const imageStateObj = useImageViewerState()

    const drawMarkOnImageAtCoord = (imageX, imageY, img) => {

        if(conservativeCellMask.getPixel(Math.floor(imageX),Math.floor(imageY))[0]!=255){
            return
        }

        const MARKER_ELEMENT = windowGet("MARKER_ELEMENT",  [[true]])

        const avgColor = annotatedImage.averageColorInsideElement(imageX,imageY,MARKER_ELEMENT)
        const invColor = [255-avgColor[0],255-avgColor[1],255-avgColor[2]]
        img.drawElement(imageX,imageY,MARKER_ELEMENT,invColor)

    }

    const markAtCoord = (imageX, imageY) => {


        if(conservativeCellMask.getPixel(Math.floor(imageX),Math.floor(imageY))[0]!=255){
            return
        }

        marks.push([Math.floor(imageX), Math.floor(imageY)])
        setMarks(marks)

        const MARKER_ELEMENT = windowGet("MARKER_ELEMENT",  [[true]])

        const avgColor = annotatedImage.averageColorInsideElement(imageX,imageY,MARKER_ELEMENT)
        const invColor = [255-avgColor[0],255-avgColor[1],255-avgColor[2]]
        annotatedImage.drawElement(imageX,imageY,MARKER_ELEMENT,invColor)
        
        imageStateObj.setImage(annotatedImage)
        imageStateObj.forceRefresh()

    }

    const eraseAtCoord = (imageX, imageY) => {


        let newMarks=[]

        let marksToDelete = []

        let erasePos = [imageX, imageY]

        const DELETE_RADIUS = windowGet("DELETE_RADIUS",0)

        marks.forEach((mark)=>{
            let D = VecOps.diff(mark, erasePos)
            let d = VecOps.magnitude(D)
            if(d > DELETE_RADIUS) {
                newMarks.push(mark)
            }else{
                marksToDelete.push(mark)
            }
        })

        marksToDelete.forEach((mark)=>{
            const MARKER_ELEMENT = windowGet("MARKER_ELEMENT",  [[true]])
            annotatedImage.restoreElement(mark[0],mark[1],MARKER_ELEMENT,image)
        })

        imageStateObj.setImage(annotatedImage)
        imageStateObj.forceRefresh()

        setMarks(newMarks)

    }

    const enableViewPristineImage = () => {


        setViewPristineImage(true)
        imageStateObj.setImage(windowGet("image",null))
        imageStateObj.forceRefresh()
        forceRefresh()
    }

    const disableViewPristineImage = () => {


        setViewPristineImage(false)
        imageStateObj.setImage(windowGet("annotatedImage",null))
        imageStateObj.forceRefresh()
        forceRefresh()
    }

    windowSet("imageFilepath",imageFilepath)
    windowSet("isSaving",isSaving)
    windowSet("isLoading",isLoading)
    windowSet("isAdjusting",isAdjusting)

    const asyncSaveMarkings = async () => {

        await new Promise((resolve, reject) => {
            setTimeout(() => {
                resolve()
            }, ASYNC_OP_MIN_DELAY_MILLIS)
        })

        // For testing the UI
        try{
            await eel.save_markings(imageFilepath,JSON.stringify(image.toPlainObject()),JSON.stringify(annotatedImage.toPlainObject()),JSON.stringify(marks))()
        }catch(e){
            setErrorMessage(formatError(e))
        }
 
        setIsSaving(false)

        forceRefresh()
    }

    const saveMarkings = () => {
        if(windowGet("imageFilepath","")){
            if(!windowGet("isSaving",false)&&!windowGet("isLoading",false)&&!windowGet("isAdjusting",false)){
                windowGet("asyncSaveMarkings",async()=>{
                    setIsSaving(false)
                    forceRefresh()
                })()
                setIsSaving(true)
            }
        }
    }

    const asyncLoadMarkings = async () => {

        await new Promise((resolve, reject) => {
            setTimeout(() => {
                resolve()
            }, ASYNC_OP_MIN_DELAY_MILLIS)
        })

        // loading logic
        const responseJSON = await eel.load_markings(imageFilepath)()
 

        if(responseJSON==="no_session"){


            setErrorMessage("No saved session.")
            setIsLoading(false)
            forceRefresh()
            return
        }

        const response = JSON.parse(responseJSON)

        let img = Image.fromJSON(response.image)
        let annotatedImg = Image.fromJSON(response.annotatedImage)
        let mrks = response.marks

        setImage(img)
        setAnnotatedImage(annotatedImg)
        setMarks(mrks)

        if(!viewPristineImage)
            imageStateObj.setImage(annotatedImg)
        else
            imageStateObj.setImage(img)
            
        imageStateObj.forceRefresh()

        setIsLoading(false)

        forceRefresh()
    }

    const loadMarkings = () => {
        if(windowGet("imageFilepath","")){
            if(!windowGet("isSaving",false)&&!windowGet("isLoading",false)&&!windowGet("isAdjusting",false)){
                windowGet("asyncLoadMarkings",async()=>{
                    setIsLoading(false)
                    forceRefresh()
                })()
                setIsLoading(true)
            }
        }
    }

    const asyncAdjustMarkings = async () => {

        await new Promise((resolve, reject) => {
            setTimeout(() => {
                resolve()
            }, ASYNC_OP_MIN_DELAY_MILLIS)
        })

        try{
            const newMarks = JSON.parse(await eel.auto_adjust_markings(JSON.stringify(greyscaleImage.toPlainObject()``),
            JSON.stringify(conservativeCellMask.toPlainObject()),JSON.stringify(marks))())

            const newAnnotatedImage = image.clone()

            newMarks.forEach((mark)=>{
                drawMarkOnImageAtCoord(Math.floor(mark[0]),Math.floor(mark[1]),newAnnotatedImage)
            })

            setAnnotatedImage(newAnnotatedImage)
            setMarks(newMarks)

            if(!viewPristineImage){
                imageStateObj.setImage(newAnnotatedImage)
            }else{
                imageStateObj.setImage(image)
            }

            imageStateObj.forceRefresh()

        }catch(e){
            setErrorMessage(formatError(e))
        }

        setIsAdjusting(false)
        forceRefresh()
    }

    const adjustMarkings = () => {
        if(windowGet("imageFilepath","")){
            if(!windowGet("isSaving",false)&&!windowGet("isLoading",false)&&!windowGet("isAdjusting",false)){
                if(marks.length > 0){
                    windowGet("asyncAdjustMarkings",async()=>{
                        setIsAdjusting(false)
                        forceRefresh()
                    })()
                    setIsAdjusting(true)
                }else{
                    setErrorMessage("No marks to adjust.")
                }
            }
        }
    }

    windowSet("asyncSaveMarkings",asyncSaveMarkings)
    windowSet("asyncLoadMarkings",asyncLoadMarkings)
    windowSet("asyncAdjustMarkings",asyncAdjustMarkings)


    const imageViewer = newImageViewer(imageStateObj,
        markAtCoord,
        eraseAtCoord,
        enableViewPristineImage,
        disableViewPristineImage,
        saveMarkings
        )

    const appInit = async () => {

        windowSet("MARKER_RADIUS",await eel.get_marker_radius()())
        windowSet("DELETE_RADIUS",await eel.get_delete_radius()())

        calculate_marker_element()

        document.addEventListener('keydown',(evt)=>{
            if(evt.key===" "){
                enableViewPristineImage()
            }
            if(evt.ctrlKey&&evt.key==="s"){
                saveMarkings()
                evt.preventDefault()
            }
            if(evt.ctrlKey&&(evt.key==="l"||evt.key==="r")){
                loadMarkings()
                evt.preventDefault()
            }
        })

        document.addEventListener('keyup',(evt)=>{
            if(evt.key===" "){
                disableViewPristineImage()
            }
        })

    }

    const imageLoadProcedure = async (filepath) => {
        const imageJSON = await eel.load_image_rpe(filepath)()
        const maskJSON = await eel.get_conservative_cell_mask_rpe(filepath)()


        const img = Image.fromJSON(imageJSON)
        const annotatedImg = img.clone()

        let mask = Image.fromJSON(maskJSON)

        const imgGs = img.toGreyscale().withMaskApplied(mask)

        setGreyscaleImage(imgGs)

        windowSet("image", img)
        windowSet("annotatedImage", annotatedImg)

        imageStateObj.setImage(img)
        imageStateObj.setOrigin([
            Math.floor(img.w/2),
            Math.floor(img.h/2)
        ])
        imageStateObj.setZoom(1.0)
        setImageFilepath(filepath)
        setImage(img)
        setConservativeCellMask(mask)
        setAnnotatedImage(annotatedImg)
        imageStateObj.show()
        forceRefresh()
    }

    const handleOpenFile = async () => {
        try{
            const filepath = await eel.pick_image()()
            if(filepath){
                await imageLoadProcedure(filepath)
            }
        }catch(e){
            setErrorMessage(formatError(e))
        }
    }

    useEffect(()=>{
        (async ()=>{
            if(!initialized){
                await appInit()
                setInitialized(true)
            }
        }
        )()
    })

    return Fragment()(Div({
        onContextMenu:(evt)=>{evt.preventDefault()}
    })(
        Nav({
            className:"navbar navbar-light bg-light navbar-expand"
        })(

            A({
                className:"navbar-brand",
                href:"#"
            })("CellMark"),

            Ul({
                className:"navbar-nav me-auto"
            })(
                A({
                    className:"nav-link",
                    href:"#",
                    onClick:handleOpenFile
                })(
                    "Open Image"
                ),
                A({
                    className:"nav-link",
                    href:"#",
                    onClick:saveMarkings
                })(
                    "Save Markings"
                ),
                isSaving?(
                    A({
                        className:"nav-link text-primary spinner-border",
                        href:"#",
                        onClick:(evt)=>{evt.preventDefault(); evt.stopPropogation(); return false;},
                        style:{
                            cursor:"default"
                        }
                    })()  
                ):(
                    Fragment()()
                ),
                A({
                    className:"nav-link",
                    href:"#",
                    onClick:loadMarkings
                })(
                    "Load Markings"
                ),
                isLoading?(
                    A({
                        className:"nav-link text-primary spinner-border",
                        href:"#",
                        onClick:(evt)=>{evt.preventDefault(); evt.stopPropogation(); return false;},
                        style:{
                            cursor:"default"
                        }
                    })()  
                ):(
                    Fragment()()
                ),
                A({
                    className:"nav-link",
                    href:"#",
                    onClick:adjustMarkings
                })(
                    "Auto-Adjust Markings"
                ),
                isAdjusting?(
                    A({
                        className:"nav-link text-primary spinner-border",
                        href:"#",
                        onClick:(evt)=>{evt.preventDefault(); evt.stopPropogation(); return false;},
                        style:{
                            cursor:"default"
                        }
                    })()  
                ):(
                    Fragment()()
                )
            )
        ),
        Div({className:"container-fluid"})(
            Div({className:"row"})(
                Div({className:"col-12",
                    style:{
                        display:"flex",
                        flexDirection:"column",
                        justifyContent:"center",
                        alignItems:"center"
                    }
                })(
                    image?imageViewer:(
                        // H3({className:"text-secondary"})(
                        //     "Get Started: ",
                        //     Code()("Open File"),
                        //     " or ",
                        //     Code()("Load Markings")
                        // )
                        Fragment()()
                    ),
                    errorMessageComponent,
                    messageComponent
                )  
            ),
        )
    ),
    // modalComponent
    )
}
