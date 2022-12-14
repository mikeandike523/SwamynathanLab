class Image{
    constructor(W,H,data=null){
        this.W=W
        this.H=H
        this.data = [];
        if(data){
            this.data=data
        }else{
            for(var i=0;i<W*H*3;i++){
                this.data.push(0)
            }
        }
    }
    static fromJSON(image_data_json){
        return new Image(image_data_json.W,image_data_json.H,image_data_json.data)
    }
    inBounds(x,y){
        return x >=0  && y>=0 && x < this.W && y < this.H
    }
    readPixel(x,y){
        if(!this.inBounds(x,y)){
            return [0,0,0]
        }
        var tid = y * this.W + x;
        return [
            this.data[tid*3+0],
            this.data[tid*3+1],
            this.data[tid*3+2]
        ]
    }
}

class Artist{
    posessContext(ctx){
        this.ctx = ctx
        return this
    }
    lock(){
        this.locked = true
        this.imageData = this.ctx.getImageData(0,0,main_canvas.width, main_canvas.height)
    }
    unlock(){
        this.ctx.putImageData(this.imageData,0,0)
        this.locked=false
    }
    readPixel(x,y){
        if(!this.locked){
            throw "Artist must be locked before reading pixels"
        }
        if(this.inBounds(x,y)){
            var tid = y * main_canvas.width + x
            return [this.imageData.data[tid*4+0],this.imageData.data[tid*4+1],this.imageData.data[tid*4+2]]
        }
        return [0,0,0]
    }
    writePixel(x,y,value){
        if(!this.locked){
            throw "Artist must be locked before writing pixels"
        }
        if(this.inBounds(x,y)){
            var tid = y * main_canvas.width + x
            for(var channel=0;channel<4;channel++){
                this.imageData.data[tid*4+channel] = channel < 3 ? value[channel] : 255
            }
        }
    }
    inBounds(x,y){
        return x>=0 && x<main_canvas.width && y>=0 && y<main_canvas.height
    }
    wash(value=[255,255,255]){
        if(!this.locked){
            throw "Artist must be locked before washing"
        }
        for(var x=0;x<main_canvas.width;x++){
            for(var y=0;y<main_canvas.height;y++){
                this.writePixel(x,y,value)
            }
        }
    }
}

function signum(value){
    if(value > 0)
        return 1
    if(value < 0)
        return -1
    return 0
}

var debounce_procedure = ()=>{}
var debounce_timer = 0
var debounce_delay = 0

function debounce(procedure, delay_millis){
    debounce_procedure = procedure
    debounce_delay = delay_millis
}

DEBOUNCE_INTERVAL = 200

function debounce_tick(){

    debounce_timer += 5
    if(debounce_timer >= debounce_delay){
        debounce_procedure()
        debounce_procedure = ()=>{}
        debounce_timer = 0
    }

}

function scalePixel(scalar,pixel){
    return [scalar*pixel[0],scalar*pixel[1],scalar*pixel[2]]
}

function addPixels(pixelA,pixelB){
    return [
        pixelA[0] + pixelB[0],
        pixelA[1] + pixelB[1],
        pixelA[2] + pixelB[2]
    ]
}

function mixPixels(weights,pixels){
    var totalPixel = [0,0,0]
    for(var i =0; i<pixels.length;i++){
        totalPixel = addPixels(totalPixel,scalePixel(weights[i],pixels[i]))
    }
    totalPixel[0] = Math.floor(totalPixel[0])
    totalPixel[1] = Math.floor(totalPixel[1])
    totalPixel[2] = Math.floor(totalPixel[2])
    return totalPixel
}

// window.setInterval(debounce_tick,DEBOUNCE_INTERVAL)

function L2Distance(A,B){
    var dx = A[0]-B[0]
    var dy = A[1]-B[1]
    return Math.sqrt(dx*dx+dy*dy)
}

function dividedBySum(arr){
    var total = 0
    for(var i=0;i<arr.length;i++){
        total+=arr[i]
    }
    var new_array = []
    if (total === 0){
        for(var i=0;i<arr.length;i++){
            new_array.push(NaN)
        }
        return new_array
    }
    for(var i=0;i<arr.length;i++){
        new_array.push(arr[i]/total)
    }
    return new_array
}

function movedDownByMin(arr){
    var minval = Math.min(...arr)
    var new_array = []
    for(var i=0;i<arr.length;i++){
        new_array.push(arr[i]-minval)
    }
    return new_array
}

function oneMinusArray(arr){
    var new_array = []
    for(var i=0;i<arr.length;i++){
        new_array.push(1.0-arr[i])
    }
    return new_array
}

class ProtectedSINC{
    constructor(epsilon=1e-6){
        this.epsilon=epsilon
    }
    wrapped(){
        return (x)=>{
            if(Math.abs(x)<this.epsilon){
                return 1.0
            }
            return Math.sin(x)/x
        }
    }
}

function showToast(message){
    $('#toastBody').text(message)
    $('#toastDiv').toast('show')
    setTimeout(()=>{
        $('#toastDiv').toast('hide')
    },1500)
    console.log(message)
}