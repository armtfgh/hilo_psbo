**HILO: Human-In-the-Loop Language-Guided Optimization**  
HILO converts plain-language expert knowledge into structured Bayesian optimization priors, then uses adaptive wrongness credibility detection (AWCD) to reduce or disable prior influence when observations conflict with that knowledge.  
This public folder contains the core runnable code and the web interface. It intentionally excludes manuscript drafts, revision figures, benchmark result CSVs, caches, and other analysis artifacts.  
**Contents**  
- readout_schema.py, prior_gp.py: core prior schema and GP-with-prior-mean utilities for continuous UGI-style domains.  
- readout_schema_p3ht.py, prior_gp_p3ht.py: P3HT-specific variants.  
- main_benchmark_portion_new_safety.py: UGI BO/AWCD benchmark implementation.  
- main_benchmark_portion_new_safety_p3ht.py: P3HT BO/AWCD benchmark implementation.  
- llm_study.py: lightweight LLM JSON-call helper for OpenAI, Anthropic, and OpenAI-compatible endpoints.  
- llm_controller.py, adaptive_cuser_runner.py: optional adaptive controller / assist-mode utilities.  
- hilo_webapp/: FastAPI + React interface for interactive HILO use.  
- ugi_merged_dataset.csv: small merged UGI lookup dataset used by the demo webapp.  
- P3HT_dataset.csv: small P3HT dataset for local examples.  
**Installation**  
Python 3.10 or 3.11 is recommended.  
cd hilo_psbo  
 python -m venv .venv  
 source .venv/bin/activate  
 pip install --upgrade pip  
 pip install -r requirements.txt  
   
The web frontend also needs Node.js 18+.  
cd hilo_webapp/frontend  
 npm install  
   
**Run The Web App**  
From the github_share root:  
cd hilo_webapp  
 ./serve.sh --rebuild  
   
Then open:  
http://localhost:8765  
   
The app includes two modes:  
- **UGI demo mode**: runs a fast lookup-oracle UGI optimization using ugi_merged_dataset.csv.  
- **Ask-Tell mode**: lets a user define their own continuous parameters, receive suggested experiments, enter measured results, and continue the optimization loop.  
**LLM Translation**  
The app can translate expert text into a structured JSON prior. This is optional: users can also directly edit JSON priors.  
For Anthropic models:  
export ANTHROPIC_API_KEY="..."  
 export ANTHROPIC_VERSION="2023-06-01"  
   
For OpenAI models:  
export OPENAI_API_KEY="..."  
   
The model registry is in llm_study.py. Update model names there if your provider uses different IDs.  
**Prior JSON Format**  
HILO priors use three primitives:  
{  
   "effects": {  
     "ptsa": {  
       "effect": "increasing",  
       "scale": 0.6,  
       "confidence": 0.8,  
       "range_hint": [0.10, 0.30]  
     }  
   },  
   "bumps": [  
     {  
       "mu": [120, 270, 300, 0.12],  
       "sigma": [20, 20, 15, 0.02],  
       "amp": 0.4  
     }  
   ],  
   "constraints": [  
     {  
       "var": "amine_mM",  
       "range": [200, 300],  
       "penalty": 8.0,  
       "reason": "high amine suppresses yield"  
     }  
   ]  
 }  
   
For the UGI demo, variables are:  
- amine_mM  
- aldehyde_mM  
- isocyanide_mM  
- ptsa  
**Run A Minimal UGI Benchmark From Python**  
import main_benchmark_portion_new_safety as ugi  
   
 domain = ugi.build_continuous_domain()  
 readout = {  
     "effects": {  
         "ptsa": {  
             "effect": "increasing",  
             "scale": 0.5,  
             "confidence": 0.7,  
             "range_hint": [0.10, 0.30],  
         }  
     },  
     "bumps": [],  
     "constraints": [  
         {  
             "var": "amine_mM",  
             "range": [200, 300],  
             "penalty": 8.0,  
             "reason": "avoid high amine",  
         }  
     ],  
 }  
   
 # See the benchmark file for campaign helper functions and plotting utilities.  
   
The webapp is the fastest way to exercise the complete interactive workflow.  
**Data Note**  
Only small runtime/example datasets are included here. Manuscript figures, revision outputs, raw benchmark sweeps, and generated result CSVs were intentionally excluded.  
**Development Notes**  
Useful commands:  
# Backend only  
 cd hilo_webapp/backend  
 uvicorn app:app --host 127.0.0.1 --port 8765  
   
 # Frontend dev server  
 cd hilo_webapp/frontend  
 npm run dev  
   
 # Production-style single server  
 cd hilo_webapp  
 ./serve.sh --rebuild  
   
Before pushing to GitHub, add your preferred LICENSE file and citation information.  
**Smoke-Tested Public Release**  
This shared folder was smoke-tested from a clean copy with:  
cd github_share/hilo_webapp  
 HILO_PORT=8877 ./serve.sh --rebuild  
   
The following checks passed:  
- Frontend build completed with npm install && npm run build.  
- The single FastAPI server served the web UI at /.  
- /api/health, /api/state, and /api/prior-surfaces returned valid responses.  
- UGI demo /api/reset followed by /api/run completed using the included ugi_merged_dataset.csv.  
- Generic ask-tell mode initialized a custom optimization, accepted measured results, and returned the next suggestions.  
