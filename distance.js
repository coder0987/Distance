const fs = require('fs');

const HEIGHT = 28;
const WIDTH = 28;
const ACTIVE = 1;

function distanceStepRemoval(imgA, imgB) {
    //Each image is a HEIGHT x WIDTH array of values 0-255
    let activeA = [];
    let activeB = [];
    let pixA = 0;
    let pixB = 0;

    //Create arrays of active pixels
    for (let r in imgA) {
        for (let c in imgB) {
            if (imgA[r][c] == ACTIVE) {
                pixA++;
                activeA.push([r,c]);
            }
            if (imgB[r][c] == ACTIVE) {
                pixB++;
                activeB.push([r,c]);
            }
        }
    }

    //For each pixel in A, find the closest active pixel in B, deactivate that pixel, and add to the sum
    let sum = 0;
    for (let i=0; i<Math.min(pixA, pixB); i++) {
        let cp = 0;
        let closest = distanceTo(activeA[i],activeB[0]);
        for (let j in activeB) {
            let current = distanceTo(activeA[i],activeB[j]);
            if (current < closest) {
                closest = current;
                cp = +j;
            }
        }
        sum += closest;
        activeB.splice(cp, 1);
    }
    return sum + Math.abs(pixA - pixB) * 2;
}

function distanceTo(a, b) {
    return Math.sqrt(Math.pow(+a[0] - +b[0], 2) + Math.pow(+a[1] - +b[1],2));
}


//Sums the active distances of each pixel, then takes the differences
//Downside: cannot detect flips along the top-left to bottom-right diagonal
function pythagoreanDistance(imgA, imgB) {
    //Top left corner
    let aDistance = 0;
    let bDistance = 0;

    for (let c = 0; c < WIDTH; c++) {
        let c2 = c * c;
        for (let r = 0; r < HEIGHT; r++) {
            if (imgA[c][r] == ACTIVE || imgB[c][r] == ACTIVE) {
                //Pixel is active
                let pd = Math.sqrt(c2 + r * r);
                if (imgA[c][r] == ACTIVE) {
                    aDistance += pd;
                }
                if (imgB[c][r] == ACTIVE) {
                    bDistance += pd;
                }
            }
        }
    }
    return Math.abs(aDistance - bDistance);
}

function printBinImg(binImg) {
    let str = '';
    for (let i in binImg[0]) {
        for (let j in binImg) {
            str += binImg[j][i] ? 'O' : ' ';
        }
        str += '\n';
    }
    console.log(str);
}

function compares() {
    //8 images in arr_imgs array
    console.log("Finding first image's closest neighbor");
    let min = 1;
    let minP = distanceStepRemoval(arr_imgs[0], arr_imgs[1]);
    for (let i=2; i<count; i++) {
        let d = distanceStepRemoval(arr_imgs[0], arr_imgs[i]);
        if (d < minP) {
            min = i;
            minP = d;
        }
    }
    console.log("Lowest: " + min);
    console.log("Distance: " + minP);
    console.log("Images:");
    printBinImg(arr_imgs[0])
    printBinImg(arr_imgs[min])
}

function test() {
    let i1 = [[255,0],[255,0]];
    let i2 = [[0,255],[255,0]];
    let srd = distanceStepRemoval(i1, i2);
    let pd = pythagoreanDistance(i1, i2);
    console.log("Step Removal Distance is " + srd);
    console.log("Pythagorean Distance is " + pd);
}

function printImage(bytes) {
    let printStr = '';
    let r = 0;
    let c = 0;
    for (b in bytes) {
        if (r == 28) {
            if (c == 28) {break;}
            r = 0;
            c++;
            printStr += '\n';
        }
        if (bytes[b]) {
            printStr += 'O';
        } else {
            printStr += ' ';
        }
        r++;
    }
    console.log(printStr);
}

function bufferToImage(bytes) {
    let img = [];
    for (let i=0; i<WIDTH; i++) {img.push([])};
    let r = 0;
    let c = 0;
    for (b in bytes) {
        if (r == 28) {
            if (c == 28) {break;}
            r = 0;
            c++;
        }
        if (bytes[b]) {
            img[r][c] = 1;
        } else {
            img[r][c] = 0;
        }
        r++;
    }
    return img;
}

function mnist(imgNum, base) {
    let data = [];
    let offset = 16 + (28*28*imgNum);
    let i = 0;
    let dataToRead = Math.ceil(28 * 28);
    fs.open('train-images.idx3-ubyte', 'r', function(err, fd) {
        if (err)
          throw err;
        const buffer = Buffer.alloc(1);
        while (i < dataToRead)
        {   
          const num = fs.readSync(fd, buffer, 0, 1, i + offset);
          if (num === 0)
            break;
          data.push(buffer[0]);
          i++;
        }
        assembleAsync(data, base);
    });
}

let count = 1;
let arr_imgs = [];
function assembleAsync(data, base) {
    if (base) {
        arr_imgs[0] = bufferToImage(data);
    } else  {
        arr_imgs[count] = bufferToImage(data);
        count++;
    }
    if (count == 800) {
        compares();
    }
}

mnist(1000, true);


for (let i = 0; i<800; i++) {
    mnist(i, false);
}

