import {fileURLToPath} from 'node:url';
import fs from 'node:fs/promises';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const DIR=fileURLToPath(new URL('../',import.meta.url)).replace(/\/$/,'');
const p=Presentation.create({slideSize:{width:1800,height:1010}});
const s=p.slides.add(); s.background.fill='#FFFFFF';
const C={ink:'#202B32',muted:'#56636E',line:'#AEB7BE',blue:'#E4EFF6',blueStroke:'#7798B1',purple:'#EAE4F4',purpleStroke:'#9D85B1',mint:'#E7F0E3',green:'#7AA477',cream:'#FBEAD0',peach:'#F5E1D7',white:'#FFFFFF',light:'#F8FAFC',red:'#B96556'};
let ix=0;
function shape(g,x,y,w,h,fill=C.white,stroke=C.ink,sw=1.5,name='shape'){
 return s.shapes.add({name:`${name}-${++ix}`,geometry:g,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:sw,style:'solid'},...(g==='roundRect'?{borderRadius:9}:{})});
}
function text(t,x,y,w,h=32,size=25,opt={}){
 const q=shape('textbox',x,y,w,h,'#FFFFFF00','#FFFFFF00',0,'label');q.text=t;
 q.text.style={typeface:opt.emoji?'Apple Color Emoji':opt.sans?'Arial':'Times New Roman',fontSize:size,color:opt.color||C.ink,bold:!!opt.bold,italic:!!opt.italic,alignment:opt.align||'center',verticalAlignment:'middle',insets:0,wrap:'none',autoFit:'none'};
 return q;
}
function sub(main,index,x,y,w,size=26){
 text(main,x,y,w,33,size,{italic:true});
 text(index,x+w*.65,y+13,w*.55,22,size*.66,{italic:true});
}
function line(pts,color=C.ink,width=1.7,dashed=false){
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);let x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x,h=Math.max(...ys)-y;
 const q=s.shapes.add({name:`route-${++ix}`,geometry:'custom',position:{left:x,top:y,width:Math.max(w,0.01),height:Math.max(h,0.01)},fill:'#FFFFFF00',line:{fill:color,width,style:dashed?'dashed':'solid'},customPaths:[{width:Math.max(w,.01),height:Math.max(h,.01),commands:pts.map((p,i)=>i?{lineTo:{x:p[0]-x,y:p[1]-y}}:{moveTo:{x:p[0]-x,y:p[1]-y}})}]});return q;
}
function poly(pts,fill,stroke=C.ink,sw=1.2){
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x,h=Math.max(...ys)-y;
 return s.shapes.add({name:`polygon-${++ix}`,geometry:'custom',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:sw},customPaths:[{width:w,height:h,commands:[...pts.map((p,i)=>i?{lineTo:{x:p[0]-x,y:p[1]-y}}:{moveTo:{x:p[0]-x,y:p[1]-y}}),{close:{}}]}]});
}
function arrow(pts,color=C.ink,width=1.8,dash=false){
 line(pts,color,width,dash);const b=pts.at(-1),a=pts.at(-2);const ang=Math.atan2(b[1]-a[1],b[0]-a[0]);const len=10,half=4.8;
 poly([b,[b[0]-len*Math.cos(ang)+half*Math.sin(ang),b[1]-len*Math.sin(ang)-half*Math.cos(ang)],[b[0]-len*Math.cos(ang)-half*Math.sin(ang),b[1]-len*Math.sin(ang)+half*Math.cos(ang)]],color,color,.2);
}
function snow(x,y,r=10){
 text('❄️',x-r-3,y-r-3,2*r+6,2*r+6,2*r+6,{emoji:true});
}
function flame(x,y,scale=1){
 text('🔥',x-2,y-2,30*scale,32*scale,32*scale,{emoji:true});
}
function module(t,x,y,w,h,fill=C.blue,icon=null,size=25){
 const q=shape('roundRect',x,y,w,h,fill,C.ink,1.6,'module');
 text(t,x+(icon?18:7),y+5,w-(icon?29:14),h-10,size);
 if(icon==='frozen')snow(x+16,y+17,9);if(icon==='trainable')flame(x+w-18,y-13,.73);return q;
}
function token(x,y,w=28,h=37,cls=false,label=''){
 shape('roundRect',x,y,w,h,cls?C.mint:C.blue,cls?C.green:C.blueStroke,1.4,'token');if(label)text(label,x,y,w,h,20,{italic:true});
}
function tokenRow(x,y,selected=false){
 token(x,y,38,38,true,'CLS');
 for(let i=0;i<(selected?3:5);i++)token(x+54+40*i,y,28,38,false,selected?'':`z${i+1}`);
 text('⋯',x+54+40*(selected?3:5),y,30,38,30);
}
function barplot(x,y,w=150,h=40,fill=C.purpleStroke){
 const v=[.31,.82,.44,.97,.61,.23];line([[x,y+h],[x+w,y+h]],C.line,1);
 for(let i=0;i<v.length;i++)shape('rect',x+i*(w/6)+3,y+h-h*v[i],w/6-7,h*v[i],fill,'#FFFFFF00',0,'distribution');
}
function grid(x,y,n=4,size=18,mask=[0,2,5,9,11,14]){
 for(let i=0;i<n*n;i++)shape('rect',x+(i%n)*size,y+Math.floor(i/n)*size,size-2.5,size-2.5,mask.includes(i)?C.purple:C.white,mask.includes(i)?C.purpleStroke:C.line,1,'mask');
}
function cat(x,y,w=158,h=122){
 shape('rect',x,y,w,h,'#EEF3F3',C.blueStroke,1.3,'fixed-image');
 shape('ellipse',x+w*.19,y+h*.53,w*.62,h*.45,'#B8B8B2','#8C9291',1,'cat-body');
 poly([[x+w*.24,y+h*.34],[x+w*.23,y+h*.06],[x+w*.43,y+h*.23]],'#C6C7C0','#7B8587',1);
 poly([[x+w*.59,y+h*.23],[x+w*.8,y+h*.06],[x+w*.78,y+h*.41]],'#C6C7C0','#7B8587',1);
 shape('ellipse',x+w*.24,y+h*.18,w*.54,h*.61,'#D6D6CE','#7B8587',1,'cat-face');
 shape('ellipse',x+w*.34,y+h*.41,w*.1,h*.08,'#91A9AB',C.ink,.8,'eye');
 shape('ellipse',x+w*.59,y+h*.41,w*.1,h*.08,'#91A9AB',C.ink,.8,'eye');
 line([[x+w*.389,y+h*.42],[x+w*.389,y+h*.48]],C.ink,1.5);line([[x+w*.639,y+h*.42],[x+w*.639,y+h*.48]],C.ink,1.5);
 poly([[x+w*.48,y+h*.54],[x+w*.55,y+h*.54],[x+w*.515,y+h*.6]],'#BE8F87','#A17872',.8);
 for(let i=0;i<3;i++){line([[x+w*.43,y+h*(.61+i*.025)],[x+w*.16,y+h*(.58+i*.08)]],'#777F80',.8);line([[x+w*.59,y+h*(.61+i*.025)],[x+w*.86,y+h*(.58+i*.08)]],'#777F80',.8);}
 for(let i=1;i<4;i++){line([[x+i*w/4,y],[x+i*w/4,y+h]],'#A7BAC4',.65);line([[x,y+i*h/4],[x+w,y+i*h/4]],'#A7BAC4',.65);}
}

// Panel framing.
text('(a)  Task-guided compression',32,21,692,43,29,{bold:true,sans:true});
text('(b)  Learning token selection',774,21,456,43,28,{bold:true,sans:true});
text('(c)  Controlled comparisons',1280,21,490,43,28,{bold:true,sans:true});
line([[747,84],[747,946]],C.line,1.4,true);line([[1252,84],[1252,946]],C.line,1.4,true);

// (a): The image and candidate classes are independent inputs.
text('Fixed image x',80,89,215,31,26,{italic:true});
s.images.add({blob:new Uint8Array(await fs.readFile(DIR+'/assets/Persian_21.jpg')),contentType:'image/jpeg',alt:'Oxford-IIIT Pet dataset: Persian_21.jpg',fit:'contain',position:{left:104,top:130,width:162,height:122}});
text('Candidate classes T',420,89,260,31,26,{italic:true});
module('Persian  /  Siamese',425,153,250,72,C.purple,null,25);
arrow([[185,252],[185,288]]);arrow([[550,225],[550,288]]);
module('CLIP image encoder\nblocks 1 … ℓ',42,288,286,78,C.blue,'frozen',26);
module('CLIP text encoder',416,288,268,78,C.blue,'frozen',26);
arrow([[185,366],[185,406]]);arrow([[550,366],[550,406]]);
tokenRow(56,410,false);
token(477,410,55,38,false,'t₁');token(555,410,55,38,false,'t₂');
text('Visual tokens Z',64,452,270,30,23,{italic:true});
text('φ(T) = [mean(T); |t₁ − t₂|]',394,458,306,30,23,{italic:true});

// Token features reach both selectors. The candidate descriptor reaches C only.
arrow([[185,485],[185,522]]);arrow([[334,429],[364,429],[364,508],[531,508],[531,522]]);
arrow([[550,448],[550,454]],C.purpleStroke);
arrow([[570,490],[570,522]],C.purpleStroke,1.5);
module('Image-only U\n[patch; CLS]',48,522,278,87,C.mint,'trainable',27);
module('Task-conditioned C\n[patch; CLS; φ(T)]',399,522,280,87,C.purple,'trainable',26);
text('wider image MLP',70,612,233,25,21,{italic:true,color:C.muted});
text('candidate-aware MLP',419,612,240,25,21,{italic:true,color:C.muted});
arrow([[185,640],[185,659]]);arrow([[539,640],[539,659]]);
module('Top-k',119,659,132,43,C.white,null,26);module('Top-k',473,659,132,43,C.white,null,26);
arrow([[185,702],[185,724]]);arrow([[539,702],[539,724]]);
tokenRow(62,725,true);tokenRow(416,725,true);
text('k patches + CLS',57,768,260,29,23);text('k patches + CLS',409,768,260,29,23);
arrow([[185,798],[185,819]]);arrow([[539,798],[539,819]]);
module('CLIP image encoder, blocks ℓ+1 … L',65,819,615,58,C.blue,'frozen',26);
text('shared weights · C and U evaluated separately',74,879,591,27,21,{italic:true,color:C.muted});
arrow([[369,908],[369,920],[438,920]]);
sub('z','U',214,907,23,26);text('or',252,909,35,30,24,{italic:true});sub('z','C',303,907,23,26);
module('Cosine over T',438,908,189,45,C.cream,null,24);arrow([[627,931],[678,931]]);text('ŷ',682,912,37,35,30,{italic:true});
// Correct text embeddings are used for classification with either selection branch.
arrow([[618,428],[712,428],[712,964],[533,964],[533,953]],C.purpleStroke,1.5);
text('T',681,701,28,24,22,{italic:true,color:C.purpleStroke});

// (b): Random-mask teacher and the common KL objective.
text('Offline utility supervision',785,95,432,34,27,{bold:true});
grid(792,155,4,20,[1,3,6,8,9,13]);grid(813,174,4,20,[0,3,4,7,10,15]);
text('Fixed-k probes',779,265,155,31,23);
arrow([[909,213],[970,213]]);
module('Frozen CLIP\nresponses',970,171,240,84,C.blue,'frozen',26);
arrow([[1090,255],[1090,306],[1005,306],[1005,324]]);
module('Ridge fit + task utilities',797,324,411,61,C.purple,null,27);
text('Training label y',792,400,167,32,23,{italic:true});
arrow([[960,415],[1005,415],[1005,385]],C.muted,1.5);
arrow([[1113,385],[1113,447]]);
barplot(1041,450,144,43);
text('Teacher target q',1003,500,214,31,24,{italic:true});
module('C / U scores',791,561,167,58,C.purple,'trainable',24);
module('KL(q ∥ softmax(s))',1009,561,207,58,C.cream,null,23);
arrow([[958,590],[1009,590]]);arrow([[1113,532],[1113,561]]);
arrow([[1113,619],[1113,645],[875,645],[875,619]],C.red,1.5,true);
text('update selector',901,651,190,25,21,{italic:true,color:C.red});

// Direct-classification continuation, preserving the frozen CLIP backbone.
line([[786,699],[1217,699]],C.line,1.1,true);
text('Direct classification continuation',779,718,447,35,25,{bold:true});
module('KL-trained\nC / U',784,780,134,75,C.purple,'trainable',24);
module('ST Top-k',954,780,121,75,C.white,null,24);
module('CLIP suffix\n+ cosine',1104,780,125,75,C.blue,null,21);snow(1116,770,8);
arrow([[918,817],[954,817]]);arrow([[1075,817],[1104,817]]);
text('Hard forward / soft backward',888,863,326,26,21,{italic:true});
module('',1010,910,193,43,C.cream,null,25);
text('ℒ',1027,913,32,33,29);text('CE',1054,930,27,20,17);text('(logits, y)',1081,914,110,33,24,{italic:true});
arrow([[1165,855],[1231,855],[1231,931],[1203,931]]);
arrow([[1010,932],[851,932],[851,855]],C.red,1.5,true);

// (c1): Cross-selection table, with task fixed within each row.
text('1  Task correspondence',1284,95,474,34,27,{bold:true,align:'left'});
text('Same image; two candidate sets',1290,140,470,29,23,{italic:true});
text('Selection',1450,178,285,27,22,{italic:true,color:C.muted});
sub('S','A',1511,212,25,27);sub('S','B',1654,212,25,27);
text('Task',1311,270,61,38,24,{italic:true});sub('T','A',1380,272,25,26);
text('Task',1311,336,61,38,24,{italic:true});sub('T','B',1380,338,25,26);
module('own',1473,258,112,55,C.purple,null,25);module('swap',1616,258,112,55,C.white,null,25);
module('swap',1473,326,112,55,C.white,null,25);module('own',1616,326,112,55,C.purple,null,25);
arrow([[1589,285],[1610,285]],C.purpleStroke,1.5);arrow([[1611,353],[1590,353]],C.purpleStroke,1.5);
text('Fix the task; exchange selected tokens',1285,399,482,30,23);

// (c2): Capacity allocation diagrams. These show network roles, not measured parameter ratios.
line([[1288,451],[1766,451]],C.line,1.1,true);
text('2  Parameter allocation',1284,470,474,34,27,{bold:true,align:'left'});
for(const [label,y] of [['C',535],['U₀',591],['U',647]])text(label,1299,y,67,43,28,{italic:true});
shape('rect',1384,535,217,43,C.blue,C.ink,1.25);shape('rect',1601,535,149,43,C.purple,C.ink,1.25);
text('image',1384,535,217,43,24);text('task',1601,535,149,43,24);
shape('rect',1384,591,217,43,C.blue,C.ink,1.25);shape('rect',1601,591,149,43,'#F0EFF2',C.ink,1.25);
text('image',1384,591,217,43,24);text('fixed task',1601,591,149,43,23);
shape('rect',1384,647,366,43,C.mint,C.ink,1.25);text('wider image MLP',1384,647,366,43,24);
text('Matched total parameter budget',1295,706,465,29,23);

// (c3): Compare learning objectives from identical existing weights.
line([[1288,750],[1766,750]],C.line,1.1,true);
text('3  Supervision objective',1284,769,474,34,27,{bold:true,align:'left'});
module('Same C / U\ncheckpoint',1288,851,178,72,C.purple,null,24);
module('Continue KL',1553,821,204,50,C.cream,null,25);
module('Direct CE',1553,903,204,50,C.cream,null,25);
arrow([[1466,887],[1505,887],[1505,847],[1553,847]]);arrow([[1505,887],[1505,928],[1553,928]]);
text('Equal additional training steps',1287,956,480,27,22,{italic:true});

// Compact legend, using vector icons and actual token glyphs.
snow(47,984,10);text('frozen',63,967,81,32,22,{align:'left'});
flame(166,970,.72);text('trainable',193,967,109,32,22,{align:'left'});
token(326,972,22,23,true);text('CLS retained',359,967,141,32,22,{align:'left'});
line([[526,972],[526,996]],C.line,1);
text('Held-out images & unseen class combinations',548,967,665,32,24,{italic:true,align:'left'});

s.speakerNotes.text=`Methodology/workflow for the DSAA 2012 ViT token compression project. Based on the current repository implementation, not a proposed unimplemented model.\n(a) Frozen CLIP prefix/suffix and text encoder; the conditioned selector uses per-patch [patch; CLS; mean(candidate text); absolute text difference]. The strong image-only selector widens its MLP to match parameter count. The CLS token is always retained, selected patches preserve spatial order, and C/U are evaluated as separate alternatives. Binary task descriptor illustrated.\n(b) Fixed-budget random subset probes and ridge regression produce task utility targets using training labels. Both selectors receive KL supervision. Later direct-CE continuation starts from KL-trained weights, freezes CLIP parameters, and backpropagates through straight-through Top-k with hard forward and soft backward behavior. Bar heights are an explanatory schematic. The input image is Persian_21.jpg from the Oxford-IIIT Pet dataset (https://www.robots.ox.ac.uk/~vgg/data/pets/), copied from the project dataset on the server. The original photo is shown without alteration.\n(c) Selection-swap evaluation fixes the target classification task within each row. Capacity comparison includes U0, the same architecture with a constant descriptor, and U, the wider image-only baseline. Parameter bands indicate architectural roles, not exact proportions. KL and CE continuations start at the same checkpoint with equal additional updates. Final evaluation uses held-out images and unseen combinations.\nLocal sources: src/token_compression/selector.py; docs/selector-data-coverage.md; docs/experiments/2026-11-08-direct-classification.md. Visual references: four paper figures provided by the user. All diagram elements and emoji markers are native editable PowerPoint shapes and text; the dataset photograph is an embedded image.`;

await fs.mkdir(DIR+'/output',{recursive:true});
await (await PresentationFile.exportPptx(p)).save(DIR+'/candidate.pptx');
const preview=await p.export({slide:s,format:'png',scale:1});
await fs.writeFile(DIR+'/preview.png',new Uint8Array(await preview.arrayBuffer()));
const layout=await s.export({format:'layout'});await fs.writeFile(DIR+'/layout.json',await layout.text());
console.log('Created candidate.pptx and preview.png; editable objects:',ix);
