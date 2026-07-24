const fs = require("fs");
const buf = fs.readFileSync("output.ppm");
let end = 0, nl = 0;
for (let i = 0; i < buf.length && nl < 3; i++) { if (buf[i] === 10) { nl++; end = i + 1; } }
const w = 512, h = 512;
const px = buf.slice(end);
let sum = 0, mn = 255, mx = 0, n = 0;
for (let i = 0; i < px.length; i += 3) {
  const v = (px[i] + px[i+1] + px[i+2]) / 3;
  sum += v; if (v < mn) mn = v; if (v > mx) mx = v; n++;
}
console.log("pixels", n, "mean", (sum/n).toFixed(1), "min", mn, "max", mx);
const g = (x, y) => { const i = (y*w + x)*3; return [px[i], px[i+1], px[i+2]]; };
console.log("center", g(256,256));
console.log("top(toward ceiling light)", g(256,40));
console.log("bottom(floor)", g(256,470));
console.log("left wall", g(20,256));
console.log("right wall", g(492,256));
