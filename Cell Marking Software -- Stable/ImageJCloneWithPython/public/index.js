psync = (new ProtectedSINC()).wrapped()

var active_image = null
var auxillary_image = null
var raw_active_image = null

var active_image_W = 0
var active_image_H = 0

var show_raw = false

var main_canvas = document.getElementById('main_canvas')
var main_canvas_ctx=main_canvas.getContext('2d',{willReadFrequenty:true})
var auxillary_canvas = document.getElementById('auxillary_canvas')
var auxillary_canvas_ctx=auxillary_canvas.getContext('2d',{willReadFrequenty:true})

var artist = new Artist().posessContext(main_canvas_ctx)
var auxilaryArtist = new Artist().posessContext(auxillary_canvas_ctx)

var origin_x = 0
var origin_y = 0
var zoom_level = 1.0

var active_plugin = null

function event_coords_wrt(evt, selector){
    var element = $(selector)
    var offsX = element.offset().left
    var offsY = element.offset().top
    var pageX = 0
    var pageY = 0
    if(evt.pageX){
        pageX = evt.pageX
        pageY = evt.pageY
    }else{
        if(evt.originalEvent.pageX){
            pageX = evt.pageX
            pageY = evt.pageY
        }else{
            throw "Could not detect any pageX property in event"
        }
    }
    return [pageX-offsX, pageY-offsY]
}

function resetPanZoom(){
    origin_x = Math.floor(main_canvas.width/2)
    origin_y = Math.floor(main_canvas.height/2)
    zoom_level = 1.0
}

min_zoom_level = 0.1
max_zoom_level = 10.0
function addZoom(delta){
    var current_zoom_level = zoom_level
    current_zoom_level+=delta
    if(current_zoom_level<min_zoom_level)
    current_zoom_level = min_zoom_level
    if(current_zoom_level>max_zoom_level)
    current_zoom_level = max_zoom_level
    return current_zoom_level
}

function map_coords(x,y){
    var init_displacement_x = x-Math.floor(main_canvas.width/2)
    var init_displacement_y = y-Math.floor(main_canvas.height/2)
    var displacement_x = init_displacement_x/zoom_level
    var displacement_y = init_displacement_y/zoom_level
    return [Math.floor(origin_x+displacement_x),Math.floor(origin_y+displacement_y)]
}

function precise_map_coords(x,y){
    var init_displacement_x = x-Math.floor(main_canvas.width/2)
    var init_displacement_y = y-Math.floor(main_canvas.height/2)
    var displacement_x = init_displacement_x/zoom_level
    var displacement_y = init_displacement_y/zoom_level
    return [origin_x+displacement_x,origin_y+displacement_y]
}


function map_coords_at_zoom(x,y,zoom){
    var init_displacement_x = x-Math.floor(main_canvas.width/2)
    var init_displacement_y = y-Math.floor(main_canvas.height/2)
    var displacement_x = init_displacement_x/zoom
    var displacement_y = init_displacement_y/zoom
    return [Math.floor(origin_x+displacement_x),Math.floor(origin_y+displacement_y)]
}

function inverse_map_coords_at_zoom(x,y,zoom){
    var displacement_x = x-origin_x
    var displacement_y = y-origin_y
    return [Math.floor(Math.floor(main_canvas.width/2)+displacement_x*zoom),Math.floor(Math.floor(main_canvas.height/2)+displacement_y*zoom)]
}


function render(){
    artist.lock()
    if(auxillary_image)
        auxilaryArtist.lock()


    // Assuming image scale from top left corner, not center
    const queryImageAtPosition = (position)=>{

        var x= position[0]
        var y= position[1]

        
        return active_image.readPixel(Math.floor(x),Math.floor(y))

        
    }

    for(var x=0;x<main_canvas.width;x++){
        for(var y=0;y<main_canvas.height;y++){

            var [mapped_x,mapped_y] = precise_map_coords(x,y)

            var floor_mapped_x = Math.floor(mapped_x)

            var floor_mapped_y = Math.floor(mapped_y)




            // Perform blending

            // R=1

            // var principle_position = [mapped_x,mapped_y]

            // var blend_positions = [];
            // for(var i=-R;i<=R;i++){
            //     for(var j=-R;j<=R;j++){
            //         blend_positions.push([floor_mapped_x+0.5+i,floor_mapped_y+0.5+j])
            //     }
            // }

            // var weights = []

            // var K = 1.0 // distance scale factor

            // for(var i=0;i<blend_positions.length;i++){
            //     weights.push(
            //         // psync(L2Distance(principle_position,blend_positions[i])/R*Math.PI)
            //         Math.exp(-Math.pow(K*L2Distance(principle_position,blend_positions[i])/R,2))
            //         )
            // }

            // weights = movedDownByMin(weights)

            // weights = dividedBySum(weights)

            // var blend_pixels = []

            // for(var i=0;i<blend_positions.length;i++){
            //     blend_pixels.push(queryImageAtPosition(blend_positions[i]))
            // }

            // var mixed_pixel = mixPixels(weights,blend_pixels)

            // artist.writePixel(x,y,mixed_pixel)

            if(!show_raw)
            artist.writePixel(x,y,active_image.readPixel(floor_mapped_x,floor_mapped_y))
            else
            artist.writePixel(x,y,raw_active_image.readPixel(floor_mapped_x,floor_mapped_y))

            if(auxillary_image){
                auxilaryArtist.writePixel(x,y,auxillary_image.readPixel(floor_mapped_x,floor_mapped_y))
            }
        }
    }
    if(auxillary_image)
        auxilaryArtist.unlock()
    artist.unlock()
}

function setConsole(messages_json_text){
    $('#plugin_output').val("")
    messages_json = JSON.parse(messages_json_text)
    console.log(messages_json)
    for(var i=0;i<messages_json.length;i++){
        $('#plugin_output').val($('#plugin_output').val()+"\n" + messages_json[i])
    }
}

function setAuxillaryImage(image_data_json_string,resetView){
    if(JSON.parse(image_data_json_string)==="none"){
        return
    }
    auxillary_image = Image.fromJSON(JSON.parse(image_data_json_string))
    auxillary_canvas.width = active_image.W
    auxillary_canvas.height = active_image.H
    $('#auxillary_canvas').css('width',active_image.W)
    $('#auxillary_canvas').css('height',active_image.H)
    if(resetView)
        resetPanZoom()
    render()
}

function setImage(image_data_json_string,resetView){
    if(JSON.parse(image_data_json_string)==="none"){
        return
    }
    active_image = Image.fromJSON(JSON.parse(image_data_json_string))
    main_canvas.width = active_image.W
    main_canvas.height = active_image.H
    $('#main_canvas').css('width',active_image.W)
    $('#main_canvas').css('height',active_image.H)
    if(resetView){
        resetPanZoom()

        // Also a super jank way of doing things
        raw_active_image = Image.fromJSON(JSON.parse(image_data_json_string))
    }
    render()
}

function loadImageByPath(path,sf){
    eel.load_image_by_path(path,sf)((data)=>{setImage(data,true)})
}

loadImageByPath("res/splash.jpeg",1.0)

$('.dropdown-toggle').dropdown()

$('#open_file').click(()=>{
    $('#auxillary_canvas').hide()
    eel.pick_file()((image_json_text)=>{
        var image_json = JSON.parse(image_json_text)
        active_image_W=image_json.W
        active_image_H=image_json.H
        setImage(image_json_text,true)
    })
})

dragging = false
start_x = 0
start_y = 0
start_origin_x = origin_x
start_origin_y = origin_y

$('#main_canvas').contextmenu(()=>false)
$('#main_canvas').mousedown((evt)=>{
    
    var [canvasX, canvasY] = event_coords_wrt(evt,'#main_canvas')
    if(evt.button===2){
        dragging = true
        start_x = evt.clientX
        start_y = evt.clientY
        start_origin_x = origin_x
        start_origin_y = origin_y
    }
    if(evt.button===0){
        if(active_plugin==="mark_cells"){
            click_coords = map_coords_at_zoom(canvasX,canvasY,zoom_level)
            coordsX =click_coords[0]
            coordsY = click_coords[1]
            console.log(`active image W: ${active_image_W}`)
            console.log(`cords X: ${coordsX}`)
            console.log(`active image H: ${active_image_H}`)
            console.log(`cords Y ${coordsY}`)
 

            if(evt.shiftKey||(evt.originalEvent&&evt.originalEvent.shiftKey)){
                eel.delete_cell_mark(coordsX,coordsY)((image_json_text)=>{
                    var [aux, main] = JSON.parse(image_json_text)
                    setAuxillaryImage(JSON.stringify(aux),false)
                    setImage(JSON.stringify(main),false)
                    eel.read_messages()((data)=>{setConsole(data)})
                })
            }
            else{
                eel.mark_cell(coordsX,coordsY)((image_json_text)=>{
                    var [aux, main] = JSON.parse(image_json_text)
                    setAuxillaryImage(JSON.stringify(aux),false)
                    setImage(JSON.stringify(main),false)
                    eel.read_messages()((data)=>{setConsole(data)})
                })
            }

        }
    }
})
$('#main_canvas').mousemove((evt)=>{
    if(dragging){
        var deltaX = evt.clientX - start_x
        var deltaY = evt.clientY - start_y
        origin_x = start_origin_x-deltaX / zoom_level
        origin_y = start_origin_y-deltaY / zoom_level
        render()
    }
    debounce(()=>{dragging=false},600)
})
$('#main_canvas').mouseup((evt)=>{
    dragging=false
})

// Fixed point zoom
// BLACK MAGIC
$('#main_canvas').bind('mousewheel',(evt)=>{
    console.log(evt.originalEvent.wheelDelta)
    var old_zoom_level = zoom_level
    var new_zoom_level = addZoom(signum(evt.originalEvent.wheelDelta)*0.1)
    var x = evt.originalEvent.clientX - $('#main_canvas').offset().left
    var y = evt.originalEvent.clientY - $('#main_canvas').offset().top
    console.log(x,y)
    var old_coords = map_coords_at_zoom(x,y,old_zoom_level)
    var new_coords = inverse_map_coords_at_zoom(old_coords[0],old_coords[1],new_zoom_level)
    var dx = new_coords[0] - x
    var dy = new_coords[1] - y
    origin_x+=dx/new_zoom_level
    origin_y+=dy/new_zoom_level
    zoom_level = new_zoom_level
    render()
    return false
})

$('#plugin_mark_cells').click(()=>{
    eel.plugin_mark_cells()((image_json_text)=>{
        active_plugin="mark_cells"
        var [aux,main] = JSON.parse(image_json_text)
        setAuxillaryImage(JSON.stringify(aux),true)
        setImage(JSON.stringify(main),true)
        eel.get_VERTICAL_MIN_RATIO()((VERTICAL_MIN_RATIO)=>{
            if(active_image_W / active_image_H >= VERTICAL_MIN_RATIO){
                $('#canvas_container').css('flex-direction','column')
            }
            else{
                $('#canvas_container').css('flex-direction','row')
            }
            $('#auxillary_canvas').width = active_image_W
            $('#auxillary_canvas').width = active_image_H
            $('#auxillary_canvas').css('width',active_image_W)
            $('#auxillary_canvas').css('height',active_image_H)
            $('#auxillary_canvas').show()
            render()
        })
    })
})

max_lines = 10

console_messages = []

for(var i=0;i<max_lines;i++){
    console_messages.push("")
}

eel.expose(pluginConsoleLog)
function pluginConsoleLog(message){
    console_messages.shift()
    console_messages.push(message)
    $('#plugin_output').text(console_message.join("\n"))
    return "ok"
}

window.moveTo(0,0)
window.resizeTo(screen.width,screen.height)

$('#plugin_mark_cells_save').click(function(){
    eel.plugin_mark_cells_save()(()=>{showToast("Annotations saved.")})
})
$('#plugin_mark_cells_reset').click(function(){
    eel.plugin_mark_cells_reset()((image_json_text)=>{
        var [aux, main] = JSON.parse(image_json_text)
        setAuxillaryImage(JSON.stringify(aux),false)
        setImage(JSON.stringify(main),false)
        showToast("Annotations reset.")
    })
})
$('#plugin_mark_cells_load').click(function(){
    eel.plugin_mark_cells_load()((image_json_text)=>{
        var image_json = JSON.parse(image_json_text)
        active_image_W=image_json.W
        active_image_H=image_json.H
        setImage(image_json_text,true)

        // super jank way of doing this
        eel.plugin_mark_cells(false)((image_json_text)=>{
            active_plugin="mark_cells"
            var [aux,main] = JSON.parse(image_json_text)
            setAuxillaryImage(JSON.stringify(aux),true)
            setImage(JSON.stringify(main),true)
            eel.get_VERTICAL_MIN_RATIO()((VERTICAL_MIN_RATIO)=>{
                if(active_image_W / active_image_H >= VERTICAL_MIN_RATIO){
                    $('#canvas_container').css('flex-direction','column')
                }
                else{
                    $('#canvas_container').css('flex-direction','row')
                }
                $('#auxillary_canvas').width = active_image_W
                $('#auxillary_canvas').width = active_image_H
                $('#auxillary_canvas').css('width',active_image_W)
                $('#auxillary_canvas').css('height',active_image_H)
                $('#auxillary_canvas').show()
                render()
            })
        })

    })
})

// --- Adapted from https://stackoverflow.com/a/56266509/5166365
document.addEventListener('keydown', function(event) {
    if(event.ctrlKey && (event.key == 's' || event.key == 'S')){
        event.preventDefault()
        eel.plugin_mark_cells_save()(()=>{showToast('Annotations saved.')})
    }
    if(!event.ctrlKey){
        if(event.key == ' '){
            show_raw = true
            render()
        }
    }
});
document.addEventListener('keyup', function(event) {
    if(!event.ctrlKey){
        if(event.key == ' '){
            show_raw = false
            render()
        }
    }
});
// ---

