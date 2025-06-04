from depth_sizing.CC_depth_sizing import pred_CC_depth
from depth_sizing.debris_depth_sizing import pred_debris_depth
from depth_sizing.fbbpf_depth_sizing import pred_FBBPF_depth
from depth_sizing.ax_circ_scrape_depth_sizing import pred_ax_circ_scrape_depth

depth_functions = {
    'nb_debris': pred_debris_depth,
    'nb_fbbpf': pred_FBBPF_depth,
    'pc_cc': pred_CC_depth,
    'nb_other': pred_ax_circ_scrape_depth,
    'pc_other': pred_ax_circ_scrape_depth
}

def predict_flaw_depth(scan, probes_data, cscans_output, config):
    
    flaw_categories = scan.categorize_flaws()

    for depth_method, flaws in flaw_categories.items():
        if depth_method not in depth_functions:
            raise ValueError(f"Unknown depth method: {depth_method}")
        depth_func = depth_functions[depth_method]
        depth_func(flaws, probes_data, cscans_output, config)
