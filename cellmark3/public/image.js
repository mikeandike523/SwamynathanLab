class Image{
    constructor(w, h, data=null){ 

        this.w = w
        this.h = h

        if(!data){
            this.data = []
            for (var i=0;i<this.w*this.h;i++){
                this.data.push([
                    Math.floor(Math.random() * 255),
                    Math.floor(Math.random() * 255),
                    Math.floor(Math.random() * 255)
                ])
            }
        }else{
            this.data = data
        }

        this.nchannels = this.data[0]?this.data[0].length:0

    }
    coordIsInBounds(x,y){
        return x >= 0 && x < this.w && y >= 0 && y < this.h
    }
    getPixel(x, y){
        if(this.coordIsInBounds(x,y))
            return this.data[x+y*this.w]
        else
            return [0,0,0]
    }
    setPixel(x, y,color){
        if(this.coordIsInBounds(x,y))
            this.data[x+y*this.w] = color
    }
    static fromJSON(jsonString){
        let jsonObj = JSON.parse(jsonString)
        let structuredData= []
        let nchannels = jsonObj.nchannels
        for (let i=0;i<jsonObj.data.length;i++){
            if(i%nchannels==0){
                structuredData.push([])
            }
            structuredData[structuredData.length-1].push(jsonObj.data[i])    
        }
        return new Image(jsonObj.w, jsonObj.h, structuredData)
    }
    clone(){
        let clone = new Image(this.w, this.h)
        clone.data = this.data.slice(0)
        return clone
    }
    averageColorInsideElement(fx,fy,element){
        let x = Math.floor(fx)
        let y = Math.floor(fy)
        let eW = element[0].length
        let eH = element.length
        let offsX = Math.floor(eW/2)
        let offsY = Math.floor(eH/2)
        let totalColor = [0,0,0]
        let denom=0
        for (let r = 0; r< eH; r++){
            for (let c = 0; c < eW; c++){
               if(element[r][c]){
                    let pixel = this.getPixel(x-offsX+c, y-offsY+r)
                    denom+=1
                    try{
                    totalColor[0]+=pixel[0]
                    totalColor[1]+=pixel[1]
                    totalColor[2]+=pixel[2]
                    }catch(e){
                        throw(e)
                    }
                }
            }
        }
        return [Math.floor(totalColor[0]/denom),Math.floor(totalColor[1]/denom),Math.floor(totalColor[2]/denom)]
    }
    drawElement(fx,fy,element,color=[0,255,0]){
        let x = Math.floor(fx)
        let y = Math.floor(fy)
        let eW = element[0].length
        let eH = element.length
        let offsX = Math.floor(eW/2)
        let offsY = Math.floor(eH/2)
        for (let r = 0; r< eH; r++){
            for (let c = 0; c < eW; c++){
               if(element[r][c]){
                this.setPixel(x-offsX+c, y-offsY+r,color)
               }
            }
        }
    }
    restoreElement(fx,fy,element,backgroundImage){
        let x = Math.floor(fx)
        let y = Math.floor(fy)
        let eW = element[0].length
        let eH = element[1].length
        let offsX = Math.floor(eW/2)
        let offsY = Math.floor(eH/2)
        for (let r = 0; r< eH; r++){
            for (let c = 0; c < eW; c++){
               if(element[r][c]){
                this.setPixel(x-offsX+c, y-offsY+r,
                        backgroundImage.getPixel(x-offsX+c, y-offsY+r)
                    )
               }
            }
        }
    }
    toPlainObject(){
        return {
            w: this.w,
            h: this.h,
            nchannels:this.nchannels,
            data:this.data
        }
    }
}