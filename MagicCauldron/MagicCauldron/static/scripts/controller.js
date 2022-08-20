


window.onload = function () {
    $('#loading').hide();
    var pasteBin = document.getElementById('pasteBin');
    var ctx = pasteBin.getContext('2d');


    var cauldron = document.getElementById('cauldron');
    var cauldTx = cauldron.getContext('2d');

    var pastedImage;



    // Make the DIV element draggable:
    dragElement(pasteBin);

    function dragElement(elmnt) {

        elmnt.style.top = "10vh";
        elmnt.style.left = "10vw";

        var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        if (document.getElementById(elmnt.id + "header")) {
            // if present, the header is where you move the DIV from:
            document.getElementById(elmnt.id + "header").onmousedown = dragMouseDown;
        } else {
            // otherwise, move the DIV from anywhere inside the DIV:
            elmnt.onmousedown = dragMouseDown;
        }

        function dragMouseDown(e) {
            e = e || window.event;
            e.preventDefault();
            // get the mouse cursor position at startup:
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            // call a function whenever the cursor moves:
            document.onmousemove = elementDrag;
        }

        function elementDrag(e) {
            e = e || window.event;
            e.preventDefault();
            // calculate the new cursor position:
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            // set the element's new position:
            elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
            elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
        }

        function closeDragElement() {
            // stop moving when mouse button is released:
            document.onmouseup = null;
            document.onmousemove = null;
        }
    }


    document.getElementById('pasteArea').onpaste = function (event) {
        // use event.originalEvent.clipboard for newer chrome versions
        var items = (event.clipboardData || event.originalEvent.clipboardData).items;
        console.log(JSON.stringify(items)); // will give you the mime types
        // find pasted image among pasted items
        var blob = null;
        for (var i = 0; i < items.length; i++) {
            if (items[i].type.indexOf("image") === 0) {
                blob = items[i].getAsFile();
            }
        }
        // load image if there is a pasted image
        if (blob !== null) {
            var reader = new FileReader();
            reader.onload = function (event) {
                console.log(event.target.result); // data url!
                // document.getElementById("pastedImage").src = event.target.result;
                var myImage = new Image();
                myImage.onload = function () {
                    pasteBin.width = myImage.width;
                    pasteBin.height = myImage.height;
                    pastedImage = myImage;
                    ctx.drawImage(myImage, 0, 0, myImage.width, myImage.height);
                    //LoadPalette();
                    ProcessImage();
                }
                myImage.src = event.target.result;
            };
            reader.readAsDataURL(blob);
        }
    }


    //Add the change listener to update image
    //inputs = document.getElementsByTagName('input');
    //for (index = 0; index < inputs.length; ++index) {
    //    inputs[index].addEventListener('change', ProcessImage);
    //}

    document.getElementById('pastescale').addEventListener('change', ProcessImage);

    StartCauldron();

    function StartCauldron() {
        cauldTx.fillRect(0, 0, cauldron.width, cauldron.height);
    }



    function ProcessImage() {

        var canvas = document.getElementById('pasteBin');
        var ctx = pasteBin.getContext('2d');
        var scale = document.getElementById('pastescale').value;


        // step 1 - resize to 50%
        var oc = document.createElement('canvas'),
            octx = oc.getContext('2d');

        oc.width = canvas.width * scale;
        oc.height = canvas.height * scale;
        octx.drawImage(canvas, 0, 0, oc.width, oc.height);

        // step 2
        octx.drawImage(oc, 0, 0, oc.width * scale, oc.height * scale);

        canvas.width = canvas.width * scale;
        canvas.height = canvas.height * scale;

        // step 3, resize to final size
        ctx.drawImage(oc, 0, 0, oc.width * scale, oc.height * scale,
            0, 0, canvas.width, canvas.height);

    }

    function LoadPalette() {
        //var checkedValue = document.querySelector('.messageCheckbox:checked').value;
        for (var i = 0; i <= 5; i++) {
            var pixel = ctx.getImageData(i * (pasteBin.width / 5), 0, 1, 1);
            var data = pixel.data;

            var rgba = `rgba(${data[0]}, ${data[1]}, ${data[2]}, ${data[3] / 255})`;
            console.log(rgba);
            var hsv = rgbToHsv(data[0], data[1], data[2]);
            hsv[0] *= 360;
            hsv[1] *= 100;
            hsv[2] *= 100;
            //console.log(hsv);
        }
    }

    var paletteData = {
        swatches: [
            {
                hue: -1,
                colors: [

                    [31, 31, 31],
                    [64, 64, 64],
                    [128, 128, 128],
                    [191, 191, 191],
                    [217, 217, 217]
                ]
            },
            {
                hue: -2,
                colors: [
                    [240, 216, 192]//skin tone,
                ]
            },
            {
                hue: -3,
                colors: [
                    [255, 255, 255],
                ]
            },
            {
                hue: -4,
                colors: [

                    [0, 0, 0]
                ]
            },
            {
                hue: 0,
                colors: [
                    [31, 0, 0],
                    [64, 13, 13],
                    [128, 46, 46],
                    [189, 94, 94],
                    [217, 146, 141]
                ]
            },
            {
                hue: 30,
                colors: [
                    [31, 15, 0],
                    [64, 39, 13],
                    [128, 87, 46],
                    [189, 142, 94],
                    [217, 179, 141]
                ]
            },
            {
                hue: 60,
                colors: [
                    [31, 27, 0],
                    [64, 60, 13],
                    [127, 127, 45],
                    [181, 189, 94],
                    [205, 217, 141]
                ]
            },
            {
                hue: 120,
                colors: [
                    [7, 31, 0],
                    [13, 64, 14],
                    [50, 128, 45],
                    [107, 189, 94],
                    [160, 217, 142]
                ]
            },
            {
                hue: 160,
                colors: [
                    [0, 31, 13],
                    [13, 64, 51],
                    [45, 128, 98],
                    [94, 189, 151],
                    [141, 217, 179]
                ]
            },
            {
                hue: 180,
                colors: [
                    [0, 26, 31],
                    [12, 58, 61],
                    [43, 116, 122],
                    [141, 189, 189],
                    [141, 217, 210]
                ]
            },
            {
                hue: 210,
                colors: [
                    [0, 20, 31],
                    [13, 38, 64],
                    [45, 93, 128],
                    [94, 157, 189],
                    [141, 199, 217]
                ]
            },
            {
                hue: 250,
                colors: [
                    [5, 0, 31],
                    [24, 12, 61],
                    [57, 45, 128],
                    [102, 94, 189],
                    [141, 141, 217]
                ]
            },
            {
                hue: 280,
                colors: [
                    [28, 0, 31],
                    [47, 13, 64],
                    [92, 45, 128],
                    [142, 94, 189],
                    [172, 141, 217]
                ]
            },
            {
                hue: 300,
                colors: [
                    [28, 0, 16],
                    [64, 13, 63],
                    [119, 45, 127],
                    [176, 94, 189],
                    [198, 141, 217]
                ]
            }
        ]
    }
};

function Stir() {


    var iterations = document.getElementById('iterations').value;

    PostImage(iterations)

}

function PostImage(i) {
    if (i == 0) {
        return;
    }
    $('#loading').show();
    var cauldron = document.getElementById('cauldron');
    var cauldTx = cauldron.getContext('2d');


    var pasteBin = document.getElementById('pasteBin');
    var pasteBinCtx = pasteBin.getContext('2d');

    var startX = parseInt(pasteBin.getBoundingClientRect().left) - parseInt(cauldron.getBoundingClientRect().left);
    var startY = parseInt(pasteBin.getBoundingClientRect().top) - parseInt(cauldron.getBoundingClientRect().top);


    var imageData = pasteBinCtx.getImageData(0, 0, pasteBin.width, pasteBin.height);


    //call its drawImage() function passing it the source canvas directly
    cauldTx.putImageData(imageData, startX, startY);

    var jpegUrl = cauldron.toDataURL("image/jpeg")

    var request =
    {
        "image": JSON.stringify(jpegUrl),
        "prompt": document.getElementById('prompt').value,
        "strength": document.getElementById('strength').value,
        "cfg": document.getElementById('cfg').value,
        "seed": document.getElementById('seed').value + i,
        "steps": document.getElementById('steps').value,
        "name": document.getElementById('name').value,
    }
    $.ajax({
        url: '/image',
        type: "POST",
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            console.log(data);
            $('#loading').hide();

            var image = new Image();
            image.onload = function () {
                cauldTx.drawImage(image, 0, 0);
                PostImage(i - 1);
            };
            var imageData = data.image.replace("b\'\"", "");
            imageData = imageData.substring(0, imageData.length - 1);

            console.log(imageData);
            image.src = imageData;
        }
    })
}


function Conjure(i) {
    if (i == 0) {
        return;
    }
    var cauldron = document.getElementById('cauldron');
    var cauldTx = cauldron.getContext('2d');

    $('#loading').show();

    var pasteBin = document.getElementById('pasteBin');
    var pasteBinCtx = pasteBin.getContext('2d');

    var request =
    {
        "prompt": document.getElementById('prompt').value,
        "strength": document.getElementById('strength').value,
        "cfg": document.getElementById('cfg').value,
        "seed": document.getElementById('seed').value,
        "steps": document.getElementById('steps').value,
        "name": document.getElementById('name').value,
    };
    $.ajax({
        url: '/conjure',
        type: "POST",
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            $('#loading').hide();
            console.log(data);

            var image = new Image();
            image.onload = function () {
                cauldTx.drawImage(image, 0, 0);
            };
            var imageData = data.image.replace("b\'\"", "");
            imageData = imageData.substring(0, imageData.length - 1);

            console.log(imageData);
            image.src = imageData;
        }
    })
}

