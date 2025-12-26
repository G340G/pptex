function val(id){ return document.getElementById(id).value; }
function chk(id){ return document.getElementById(id).checked; }

function yamlEscapeList(csv){
  return csv.split(",").map(s=>s.trim()).filter(Boolean);
}

function build(){
  const cfg = {
    seed: Number(val("seed")),
    output: "out.mp4",
    format: { width: 640, height: 480, fps: 15, aspect: "4:3" },
    story: {
      slide_count: Number(val("slide_count")),
      slide_sec: 3.0,
      include_wellness_ratio: 0.35,
      keywords: yamlEscapeList(val("keywords")),
      encrypt_words: yamlEscapeList(val("encrypt_words")),
      cipher: val("cipher"),
      jane_doe_frequency: 0.18,
      fatal_frequency: 0.12
    },
    sources: {
      use_local_images: true,
      use_local_faces: true,
      use_web_scrape: chk("use_web"),
      web: {
        provider: "wikimedia",
        search_terms: yamlEscapeList(val("scrape_terms")),
        max_download: Number(val("max_download")),
        min_width: Number(val("min_width")),
        license_allow: ["public domain","cc by","cc by-sa","cc0"]
      }
    },
    effects: {
      enable_vhs: chk("vhs"),
      enable_glitch: chk("glitch"),
      enable_redactions: chk("redactions"),
      enable_flashes: chk("flashes"),
      enable_popups: chk("popups"),
      intensity: Number(val("intensity")),
      ppt90s_palette: chk("ppt90s"),
      ppt90s_wordart: chk("wordart")
    },
    audio: {
      enable_tts: true,
      voice: "en-us",
      tts_speed: 152,
      tts_pitch: 32,
      tts_amp: 170,
      weird_voice_fx: true,
      harsh_bed_level: 0.12,
      pop_count: 22
    }
  };

  // quick YAML-ish stringify (simple)
  const y = (obj, indent=0) => {
    const pad = "  ".repeat(indent);
    if (Array.isArray(obj)) return `[${obj.map(v => JSON.stringify(v)).join(", ")}]`;
    if (obj && typeof obj === "object"){
      return Object.entries(obj).map(([k,v]) => {
        if (v && typeof v === "object" && !Array.isArray(v)){
          return `${pad}${k}:\n${y(v, indent+1)}`;
        }
        return `${pad}${k}: ${Array.isArray(v) ? y(v) : (typeof v==="string"? JSON.stringify(v): v)}`;
      }).join("\n");
    }
    return `${obj}`;
  };

  document.getElementById("out").value = y(cfg);

  document.getElementById("inputs").value =
`seed: ${cfg.seed}
slide_count: ${cfg.story.slide_count}
intensity: ${cfg.effects.intensity}
keywords: ${cfg.story.keywords.join(",")}
encrypt_words: ${cfg.story.encrypt_words.join(",")}
scrape_terms: ${cfg.sources.web.search_terms.join(",")}
use_web_scrape: ${cfg.sources.use_web_scrape}`;
}

function copyFrom(id){
  const el = document.getElementById(id);
  el.select();
  document.execCommand("copy");
}

["seed","slide_count","keywords","encrypt_words","cipher","intensity","scrape_terms","max_download","min_width"]
  .forEach(id => document.getElementById(id).addEventListener("input", build));

["vhs","glitch","redactions","flashes","popups","ppt90s","wordart","use_web"]
  .forEach(id => document.getElementById(id).addEventListener("change", build));

document.getElementById("copyCfg").onclick = () => copyFrom("out");
document.getElementById("copyInputs").onclick = () => copyFrom("inputs");

build();
