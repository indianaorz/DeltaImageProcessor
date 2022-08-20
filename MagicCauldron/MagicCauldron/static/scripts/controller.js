


window.onload = function () {

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
    inputs = document.getElementsByTagName('input');
    for (index = 0; index < inputs.length; ++index) {
        inputs[index].addEventListener('change', ProcessImage);
    }


    StartCauldron();

    function StartCauldron() {
        cauldTx.fillRect(0, 0, cauldron.width, cauldron.height);
    }



    function ProcessImage() {


        var currentPalette = [];
        var inputElements = document.getElementsByClassName('messageCheckbox');
        for (var i = 0; inputElements[i]; ++i) {
            if (inputElements[i].checked) {
                currentPalette.push(paletteData.swatches.filter(p => p.hue == inputElements[i].value)[0]);
            }
        }




        ctx.drawImage(pastedImage, 0, 0, pastedImage.width, pastedImage.height);

        const imageData = ctx.getImageData(0, 0, pasteBin.width, pasteBin.height);
        var dat = imageData.data;
        var specialColors = [];

        currentPalette.forEach(swatch => {
            if (swatch.hue < -1) {
                swatch.colors.forEach(c => {
                    specialColors.push(c);
                });
            }

            swatch.weight = document.getElementById(swatch.hue + "Weight")?.value;
            swatch.sat = document.getElementById(swatch.hue + "Sat")?.value

        });

        var darken = document.getElementById("Darken").value;
        var currentSwatch = { colors: [] };
        for (let i = 0; i < dat.length; i += 4) {
            var hsv = rgbToHsv(dat[i], dat[i + 1], dat[i + 2]);
            var hue = hsv[0] * 360;
            var tempSwatchPointer;


            var value = hsv[2] * 100;
            var currentColor;
            var minValueDifference = 100000000;

            currentSwatch.colors = [];

            var minHueDistance = 100000000;

            var complete = false;
            //find closest avail color
            currentPalette.forEach(swatch => {
                if (complete) {
                    return;
                }
                var distance = Math.min(Math.abs(swatch.hue - hue), 360 - Math.abs(swatch.hue - hue));

                //white
                if (hsv[2] > swatch.weight && swatch.hue == -3
                    && hsv[1] < swatch.sat) {
                    tempSwatchPointer = swatch;
                    complete = true;
                }

                // //Skin
                // if (hsv[2] > swatch.weight&& swatch.hue == -2
                // && hsv[1] < swatch.sat) {
                //     tempSwatchPointer = swatch;
                //     complete = true;
                // }

                //black
                if (hsv[2] < swatch.weight && swatch.hue == -4
                    && hsv[1] < swatch.sat) {
                    tempSwatchPointer = swatch;
                    complete = true;
                }


                //desaturated
                if (hsv[1] < swatch.weight && swatch.hue == -1) {
                    tempSwatchPointer = swatch;
                    complete = true;
                }

                var weight = swatch.weight;
                if (weight != undefined) {
                    distance /= weight;
                }
                if (distance < minHueDistance
                    && swatch.hue >= 0) {
                    tempSwatchPointer = swatch;
                    minHueDistance = distance;
                }
            });

            if (tempSwatchPointer != null) {
                tempSwatchPointer.colors.forEach(c => {
                    currentSwatch.colors.push(c);
                });
            }

            // specialColors.forEach(c => {
            //     currentSwatch.colors.push(c);
            // });

            //find closest color on that palette
            currentSwatch.colors.forEach(color => {
                var swatchHSV = rgbToHsv(color[0], color[1], color[2]);


                var distance = Math.abs(swatchHSV[2] * 100 - value);
                if (distance < minValueDifference) {
                    currentColor = color;
                    minValueDifference = distance;
                }
            });



            var index = currentSwatch.colors.indexOf(currentColor);
            if (darken != 0) {
                var newIndex = +index + +darken;
                newIndex = Math.max(0, newIndex);
                newIndex = Math.min(currentSwatch.colors.length - 1, newIndex);
                currentColor = currentSwatch.colors[newIndex];
            }

            dat[i] = currentColor[0];
            dat[i + 1] = currentColor[1];
            dat[i + 2] = currentColor[2];
        }
        ctx.putImageData(imageData, 0, 0);
        //ctx.drawImage(img, 0, 0);

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
    var cauldron = document.getElementById('cauldron');
    var cauldTx = cauldron.getContext('2d');


    var pasteBin = document.getElementById('pasteBin');
    var pasteBinCtx = pasteBin.getContext('2d');

    var startX = parseInt(pasteBin.getBoundingClientRect().left) - parseInt(cauldron.getBoundingClientRect().left);
    var startY = parseInt(pasteBin.getBoundingClientRect().top) - parseInt(cauldron.getBoundingClientRect().top);




    var imageData = pasteBinCtx.getImageData(0, 0, pasteBin.width, pasteBin.height);

    //call its drawImage() function passing it the source canvas directly
    cauldTx.putImageData(imageData, startX, startY);
    var jpegUrl = cauldron.toDataURL();

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/image", true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify(jpegUrl));

}
