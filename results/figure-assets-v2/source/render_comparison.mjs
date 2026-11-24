import {fileURLToPath} from 'node:url';
import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const DIR=fileURLToPath(new URL('../',import.meta.url)).replace(/\/$/,'');
const ASSETS=DIR+'/comparison-assets';
const data=JSON.parse(await fs.readFile(ASSETS+'/score_examples.json','utf8'));
const p=Presentation.create({slideSize:{width:1860,height:1400}});
const s=p.slides.add();s.background.fill='#FFFFFF';
const ink='#1E2931',muted='#5E6870',blue='#3677B5',orange='#CF7132';let n=0;
function rect(x,y,w,h,fill='#FFFFFF00',stroke='#FFFFFF00',width=0){return s.shapes.add({name:'region-'+(++n),geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width}});}
function text(t,x,y,w,h=34,size=24,o={}){const q=rect(x,y,w,h);q.text=t;q.text.style={typeface:o.sans?'Arial':'Times New Roman',fontSize:size,color:o.color||ink,bold:!!o.bold,italic:!!o.italic,alignment:o.align||'left',verticalAlignment:'middle',insets:0,wrap:'none',autoFit:'none'};return q;}
function wrap(t,len=28){const out=[];let cur='';for(const w of t.split(' ')){if(cur.length+w.length+1>len){out.push(cur);cur=w;}else cur+=(cur?' ':'')+w;}if(cur)out.push(cur);return out.join('\n');}
function title(t){return t.replace(/\b\w/g,c=>c.toUpperCase());}
const cols=[330,578,826,1074,1322,1570];
const names=['Full CLIP','Attention','PR (adapted)','Shared U','C: task A','C: task B'];
text('CLIP ViT-B/16     block 6     98 / 196 patch tokens retained     KL-trained selectors     seed 2041',30,12,1770,29,22,{color:muted,align:'right'});
text('Task context',30,61,275,40,29,{bold:true,sans:true});
for(let i=0;i<6;i++)text(names[i],cols[i]-6,61,236,40,28,{bold:true,sans:true,align:'center',color:i===4?blue:i===5?orange:ink});
const methods=[null,'Attention','PR','U','C_A','C_B'];
for(let r=0;r<data.cases.length;r++){
 const c=data.cases[r], y=130+r*310, bytes=new Uint8Array(await fs.readFile(path.join(ASSETS,c.clip_crop_filename)));
 const A=new Set(c.methods.C_A.selected_indices),B=new Set(c.methods.C_B.selected_indices), changed=[...A].filter(i=>!B.has(i)).length;
 if(r)rect(30,y-18,1764,1,'#C6CDD2');
 text(title(c.true_class_name),30,y,275,36,27,{bold:true});
 text(c.dataset==='pets'?'Oxford Pets':'Flowers102',30,y+36,275,26,20,{color:muted,italic:true});
 text(c.figure_category||c.selection_category||'Selection example',30,y+67,285,29,21,{bold:true});
 for(let j=0;j<2;j++){
  const task=c.tasks[j];const competitor=task.candidate_names[task.candidate_ids.findIndex(i=>i!==c.true_class_id)];
  text(`${task.name}: vs. ${wrap(title(competitor),24)}`,30,y+102+j*58,277,49,21,{color:j?orange:blue});
 }
 text(`${changed} / 98 selections change`,30,y+224,280,28,21,{italic:true});
 if(c.figure_note||c.figure_percentile)text(c.figure_note||c.figure_percentile,30,y+253,290,26,19,{color:muted,italic:true});
 for(let k=0;k<6;k++){
  const x=cols[k];s.images.add({blob:bytes,contentType:'image/png',alt:`${c.dataset} ${c.image_id}: ${names[k]}`,fit:'contain',position:{left:x,top:y,width:224,height:224}});
  if(methods[k]){
   const selected=new Set(c.methods[methods[k]].selected_indices);
   if(selected.size!==98)throw new Error('Invalid mask size');
   for(let z=0;z<196;z++)if(!selected.has(z))rect(x+(z%14)*16,y+Math.floor(z/14)*16,16,16,'#EDEDEDF2');
   if(k>=4){const other=k===4?B:A;for(const z of selected)if(!other.has(z))rect(x+(z%14)*16+.8,y+Math.floor(z/14)*16+.8,14.4,14.4,'#FFFFFF00',k===4?blue:orange,1.65);}
  }
  rect(x,y,224,224,'#FFFFFF00','#9BA8B0',.8);
  const scores=c.methods[methods[k]||'Full'].classification_by_task;
  if(k<4){
   for(const [j,t] of ['A','B'].entries())text(`${t}: ${scores[t].correct?'correct':'wrong'}`,x+j*112,y+229,112,25,20,{color:scores[t].correct?'#376852':'#B53F3B',align:'center'});
  }else{
   const t=k===4?'A':'B',q=scores[t],delta=q.geometric_margin-c.methods.U.classification_by_task[t].geometric_margin;
   text(`${t}: ${q.correct?'correct':'wrong'}`,x,y+229,224,25,21,{color:q.correct?'#376852':'#B53F3B',align:'center'});
   text(`Δm = ${delta>=0?'+':''}${delta.toFixed(3)}`,x,y+257,224,24,21,{align:'center',italic:true});
  }
 }
}
text('Each task pairs the named class with its listed competitor. Gray: pruned patches. Outlines: selected only for A (blue) or B (orange).',30,1346,1770,29,22);
text('Δm = geometric margin of C − U on the same task. Cases illustrate typical changes, a corrected error and a large-change failure.',30,1373,1770,24,21,{italic:true,color:muted});
s.speakerNotes.text=`Real token-selection visualizations from existing final-test outputs, not illustrations of inferred masks. Settings: frozen CLIP ViT-B/16, after block 6, 98/196 patch tokens, seed 2041 for trained selectors. Source: score_examples.json accompanying this artifact. The Full CLIP column shows the complete input crop and outcomes from the uncompressed model.\nEach original image is processed with the existing CLIP preprocessing from the server: bicubic short-side resize to224, center crop224 and RGB conversion. The 14x14 mask is aligned to that exact crop. Each selected set has98 valid patch indices and agrees with saved scores' top98. PR/U are image-only and identical for task A/B. PR is the project's adapted Preservation predictor, using its development-selected checkpoint. C/U here are the KL-trained checkpoints from final-test-v1.\nCases are selected by stated behavior categories after analyzing the complete final-test pool. Typical refers to replacement count, not a claim that the individual performance difference equals the population average. See selection_summary.json and score_examples.json for full-pool distributions, image and task identifiers, and category criteria. Two median-like examples illustrate typical selection change; a helpful example illustrates an error corrected relative to U; a large-change example illustrates a boundary of the benefit. Displayed outcomes and C-U differences always compare methods within the same task. Seed2041 is used for all displayed masks; other existing seeds are summarized in the source metadata.\nVisual reference: Patch Ranking Figure3, https://arxiv.org/html/2409.14607v2#S3.F3. Images: Oxford-IIIT Pet dataset https://www.robots.ox.ac.uk/~vgg/data/pets/ and Flowers102 https://www.robots.ox.ac.uk/~vgg/data/flowers/102/.\nDataset image IDs: ${data.cases.map(c=>c.image_id).join(', ')}.`;
await (await PresentationFile.exportPptx(p)).save(DIR+'/representative-candidate.pptx');
const png=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(DIR+'/representative-preview.png',new Uint8Array(await png.arrayBuffer()));
console.log('Rendered four real-image cases.');
