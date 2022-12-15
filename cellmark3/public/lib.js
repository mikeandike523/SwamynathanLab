const R = {}

R.cE = React.createElement

useState = React.useState
useEffect = React.useEffect
useRef = React.useRef

function Div(props) {
    return (...children)=>{return R.cE("div", props,...children)}
}
function Nav(props) {
    return (...children)=>{return R.cE("nav", props,...children)}
}
function A(props) {
    return (...children)=>{return R.cE("a", props,...children)}
}
function Ul(props) {
    return (...children)=>{return R.cE("ul", props,...children)}
}
function Li(props) {
    return (...children)=>{return R.cE("ul", props,...children)}
}
function Textarea(props) {
    return (...children)=>{return R.cE("textarea", props,...children)}
}
function Button(props) {
    return (...children)=>{return R.cE("button", props,...children)}
}
function Br(props) {
    return (...children)=>{return R.cE("br", props,...children)}
}
function Span(props) {
    return (...children)=>{return R.cE("span", props,...children)}
}
function Hr() {
    return (...children)=>{return R.cE('hr',{},...children)}
}
function Code() {
    return (...children)=>{return R.cE('code',{},...children)}
}
function H1() {
    return (...children)=>{return R.cE('h1',{},...children)}
}
function H2() {
    return (...children)=>{return R.cE('h2',{},...children)}
}
function H3() {
    return (...children)=>{return R.cE('h3',{},...children)}
}
function H4() {
    return (...children)=>{return R.cE('h4',{},...children)}
}
function H5() {
    return (...children)=>{return R.cE('h5',{},...children)}
}
function H6() {
    return (...children)=>{return R.cE('h6',{},...children)}
}
function P() {
    return (...children)=>{return R.cE('h6',{},...children)}
}
function Fragment() {
    return (...children)=>{return R.cE(React.Fragment,{},...children)}
}

function toPlainObject(obj, references=[]){
    if(!obj){
        return "(nullish)"
    }
    if(typeof obj === "string") return obj
    if(typeof obj !== "object") return obj
    var plainObject = {}
    Object.getOwnPropertyNames(obj).forEach((propertyName)=>{
        if(obj[propertyName]){
        if(references.includes(obj[propertyName])){
            plainObject[propertyName] = "(circular)"
        }else{
            if(typeof obj[propertyName] === "string")
                plainObject[propertyName] = obj[propertyName]
            else{
                if(typeof obj[propertyName] === "object")
                {
                    references.push(obj[propertyName])
                    plainObject[propertyName] = toPlainObject(obj[propertyName], references)
                }
                else{
                    plainObject[propertyName] = obj[propertyName]
                }
            }
        }}else{
            plainObject[propertyName] = "(nullish)"
        }
    })
    return plainObject
}

class Debug {
    static output(obj, color="black") {
        var plainObj = toPlainObject(obj)
    }
    static log(obj) {
        Debug.output(obj,"black")
    }
    static info(obj) {
        Debug.output(obj,"blue")
    }
    static error(obj) {
        Debug.output(obj,"red")
    }
    static warn(obj) {
        Debug.output(obj,"yellow")
    }
    static success(obj) {
        Debug.output(obj,"green")
    }
}

function formatMessage(m, spaces=2) {
    let plainObject = toPlainObject(m)
    return JSON.stringify(plainObject, null, spaces)
}

function formatError(e,spaces=2){ 
    let plainObject = toPlainObject(e)
    if(plainObject.hasOwnProperty("errorText")&&plainObject.hasOwnProperty("errorTraceback")){
        return `
Error Text:
${plainObject.errorText}
Error Traceback:
${plainObject.errorTraceback}
        `
    }
    return JSON.stringify(plainObject,null,spaces)
}

class DoOnce{
    constructor(procedure){
        this.procedure = procedure
        this.hasBeenDone = false
    }
    do(args){
        if(!this.hasBeenDone){
            this.hasBeenDone = true
            this.procedure(args)
        }
    }
}

class Datastore {
    constructor() {
        this.datastore = {}
    }
    get(key) {
        if(this.datastore.hasOwnProperty(key)){
            return this.datastore[key]
        }else{
            throw new Error(`key ${key} does not exist in datastore.`);
        }
    }
    coalesce(key, defaultValue) {
        if(this.datastore.hasOwnProperty(key)){
            return this.datastore[key]
        }else{
            return defaultValue
        }
    }
    set(key, value) {
        this.datastore[key] = value;
    }
    update(key, value){
        if(this.datastore.hasOwnProperty(key)) {
            this.set(key, value)
        }else{
            throw new Error(`Key ${key} does not exist in the datastore. To set the value regardless, \`use datastore.set(...)\``)
        }
    }
    remove(key){
        if(this.datastore.hasOwnProperty(key)) {
            delete this.datastore[key]
        }else{
            throw new Error(`key ${key} does not exist in datastore.`);
        }
    }
    attemptRemove(key){
        if(this.datastore.hasOwnProperty(key)) {
            delete this.datastore[key]
        }
    }
}

function signum(value) {
    if(value>0) return 1;
    if(value<0) return -1;
    return 0;
}

function clipValue(value, min, max){
    if(value>max) return max;
    if(value<min) return min;
    return value;
}

function useModal(){
    const [open, setOpen] = useState(false)
    const [title, setTitle] = useState(null)
    const [body, setBody] = useState(null)

    const modalComponent = R.cE(function(props){
        return Div({className:"modal",
            style:{
                display:(open?"block":"hidden")
            }
        })(
            Div({className:"modal-dialog"})(
                Div({className:"modal-header"})(
                    Div({className:"modal-title"})(
                        title
                    ),
                    Button({type:"button", className:"btn-close", onClick:()=>{
                        setOpen(false)
                    }})()
                ),
                Div({className:"modal-body"})(
                    body
                )
            )
        )
    })
    const showModal=(title,body)=>{
        setTitle(title)
        setBody(body)
        setOpen(true)
    }

    return [modalComponent, showModal]

}

// --- Courtesy of https://stackoverflow.com/a/6234804/5166365
function escapeHtml(unsafe)
{
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
 }
// ---

function textToDiv(str, props={}){
    var strs = str.split("\n")
    var elements = []
    for(var i=0;i<strs.length;i++){
        elements.push(Span()(
            escapeHtml(strs[i])
        ))
        if(i<strs.length-1){
            elements.push(Br()())
        }
    }
    return Div(props)(elements)
}

class VecOps{

    static zeros(len){
        let V = []
        for(let i=0;i<len;i++){
            V.push(0)
        }
        return V
    }

    static sum(A,B){
        if(B!==undefined){
            let len = A.length
            let C = VecOps.zeros(len)
            for(let idx = 0;idx < len;idx++){
                C[idx] = A[idx] + B[idx]
            }
            return C
        }else{
            let len = A.length
            let total = 0
            for(let idx = 0;idx < len;idx++){
                total += A[idx]
            }
            return total
        }
    }

    static diff(A,B){
        let len = A.length
        let C = VecOps.zeros(len)
        for(let idx = 0;idx < len;idx++){
            C[idx] = A[idx] - B[idx]
        }
        return C
    }

    static tbtMul(A,B){
        let len = A.length
        let C = VecOps.zeros(len)
        for(let idx = 0;idx < len;idx++){
            C[idx] = A[idx] * B[idx]
        }
        return C
    }

    static dot(A,B){
        let len = A.length
        let total = 0.0
        for(let idx = 0;idx < len;idx++){
            total += A[idx] * B[idx]
        }
        return total
    }

    static magnitude(A){
        return Math.sqrt(VecOps.dot(A,A))
    }

}

function windowAppStateInit(){
    if(!window.CellMarkState){
        window.CellMarkState = {}
    }
}
function windowSet(key,value){
    windowAppStateInit()
    window.CellMarkState[key] = value 
}
function windowGet(key,defaultValue=undefined){
    windowAppStateInit()
    return window.CellMarkState[key] ?? defaultValue
}


