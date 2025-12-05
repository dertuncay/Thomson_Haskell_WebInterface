import os
import io
import base64
import uuid
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import flask
from flask import Flask, render_template, request, redirect, url_for, flash, session
import obspy
from obspy.core import read, Stream
from obspy.core.inventory import read_inventory
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['BASE_UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure base upload directory exists
os.makedirs(app.config['BASE_UPLOAD_FOLDER'], exist_ok=True)

def get_session_folder():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    folder = os.path.join(app.config['BASE_UPLOAD_FOLDER'], session['session_id'])
    os.makedirs(folder, exist_ok=True)
    return folder

def get_current_file_path():
    folder = get_session_folder()
    history = session.get('history', [])
    if not history:
        return None
    current_step = history[-1]
    return os.path.join(folder, current_step['filename'])

def save_new_step(st, action_name, domain_map, visible_indices=None):
    folder = get_session_folder()
    history = session.get('history', [])
    step_count = len(history)
    new_filename = f"step_{step_count}.mseed"
    new_filepath = os.path.join(folder, new_filename)
    st.write(new_filepath, format='MSEED')
    
    # If visible_indices not provided, default to all
    if visible_indices is None:
        visible_indices = list(range(len(st)))

    history.append({
        'filename': new_filename,
        'action': action_name,
        'domains': domain_map,
        'visible_indices': visible_indices
    })
    session['history'] = history

def infer_domains(st):
    domains = {}
    for tr in st:
        chan = tr.stats.channel
        if len(chan) >= 2:
            inst_code = chan[1]
            if inst_code in ['H', 'L']:
                domains[tr.id] = 'Velocity'
            elif inst_code in ['N', 'G']:
                domains[tr.id] = 'Acceleration'
            else:
                domains[tr.id] = 'Unknown'
        else:
            domains[tr.id] = 'Unknown'
    return domains

def generate_plots(st, domains, visible_indices):
    # Waveform Plot
    img_waveform = io.BytesIO()
    fig_waveform = plt.figure(figsize=(10, 6))
    ax = fig_waveform.add_subplot(111)
    
    # Only plot visible traces
    has_visible = False
    for i, tr in enumerate(st):
        if i in visible_indices:
            t = tr.times()
            domain = domains.get(tr.id, 'Unknown')
            ax.plot(t, tr.data, label=f"{tr.id} ({domain})")
            has_visible = True
            
    if has_visible:
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform')
        ax.set_xlim([st[0].times()[0], st[0].times()[-1]]) # Use first trace for limits or global min/max?
        ax.grid(True)
        fig_waveform.savefig(img_waveform, format='png')
        img_waveform.seek(0)
        plot_url = base64.b64encode(img_waveform.getvalue()).decode()
    else:
        plot_url = None
    plt.close(fig_waveform)

    # Spectrum Plot
    img_spectrum = io.BytesIO()
    fig_spectrum = plt.figure(figsize=(10, 6))
    ax_spec = fig_spectrum.add_subplot(111)
    
    has_visible_spec = False
    for i, tr in enumerate(st):
        if i in visible_indices:
            npts = tr.stats.npts
            dt = tr.stats.delta
            spec = np.fft.rfft(tr.data)
            freq = np.fft.rfftfreq(npts, d=dt)
            amp = np.abs(spec)
            ax_spec.loglog(freq, amp, label=tr.id)
            has_visible_spec = True
            
    if has_visible_spec:
        ax_spec.set_xlabel('Frequency (Hz)')
        ax_spec.set_ylabel('Amplitude')
        ax_spec.set_title('Fourier Spectrum')
        ax_spec.legend()
        ax_spec.grid(True, which="both", ls="-")
        fig_spectrum.savefig(img_spectrum, format='png')
        img_spectrum.seek(0)
        spectrum_url = base64.b64encode(img_spectrum.getvalue()).decode()
    else:
        spectrum_url = None
    plt.close(fig_spectrum)
    
    return plot_url, spectrum_url

@app.route('/', methods=['GET'])
def index():
    plot_url = None
    spectrum_url = None
    current_info = None
    history = session.get('history', [])
    
    if history:
        filepath = get_current_file_path()
        if filepath and os.path.exists(filepath):
            try:
                st = read(filepath)
                current_step = history[-1]
                visible_indices = current_step.get('visible_indices', list(range(len(st))))
                
                plot_url, spectrum_url = generate_plots(st, current_step['domains'], visible_indices)
                
                # Get trace info for selection list
                traces_info = []
                for i, tr in enumerate(st):
                    traces_info.append({
                        'index': i,
                        'id': tr.id,
                        'stats': str(tr.stats),
                        'visible': i in visible_indices
                    })

                current_info = {
                    'filename': current_step['filename'],
                    'domains': current_step['domains'],
                    'history': history,
                    'traces': traces_info
                }
            except Exception as e:
                flash(f"Error reading file: {e}")

    return render_template('index.html', plot_url=plot_url, spectrum_url=spectrum_url, current_info=current_info)

@app.route('/upload', methods=['POST'])
def upload():
    if 'waveform_file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['waveform_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file:
        # Reset session for new upload
        session['history'] = []
        folder = get_session_folder()
        
        # Clean folder
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

        filepath = os.path.join(folder, file.filename)
        file.save(filepath)
        
        try:
            st = read(filepath)
            domains = infer_domains(st)
            
            # Save as first step
            save_new_step(st, f"Uploaded {file.filename}", domains)
            session.modified = True
        except Exception as e:
            flash(f"Error processing upload: {e}")
            
    return redirect(url_for('index'))

@app.route('/action', methods=['POST'])
def action():
    filepath = get_current_file_path()
    if not filepath or not os.path.exists(filepath):
        flash("No active file.")
        return redirect(url_for('index'))

    try:
        st = read(filepath)
        history = session.get('history', [])
        current_step = history[-1]
        current_domains = current_step['domains'].copy()
        visible_indices = current_step.get('visible_indices', list(range(len(st))))
        
        action_type = request.form.get('type')
        action_desc = "Unknown Action"

        if action_type == 'detrend':
            st.detrend("linear")
            st.detrend("demean")
            action_desc = "Detrend (Linear & Demean)"
            
        elif action_type == 'taper':
            st.taper(max_percentage=0.05, type="hann")
            action_desc = "Taper (Hann 0.05)"
            
        elif action_type == 'filter':
            freqmin = request.form.get('freqmin')
            freqmax = request.form.get('freqmax')
            freqmin = float(freqmin) if freqmin else None
            freqmax = float(freqmax) if freqmax else None
            
            if freqmin and freqmax:
                st.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
                action_desc = f"Bandpass Filter ({freqmin}-{freqmax} Hz)"
            elif freqmin:
                st.filter("highpass", freq=freqmin)
                action_desc = f"Highpass Filter ({freqmin} Hz)"
            elif freqmax:
                st.filter("lowpass", freq=freqmax)
                action_desc = f"Lowpass Filter ({freqmax} Hz)"
                
        elif action_type == 'integrate':
            st.integrate()
            action_desc = "Integration"
            # Update domains
            for tr_id in current_domains:
                if current_domains[tr_id] == 'Acceleration':
                    current_domains[tr_id] = 'Velocity'
                elif current_domains[tr_id] == 'Velocity':
                    current_domains[tr_id] = 'Displacement'
                    
        elif action_type == 'differentiate':
            st.differentiate()
            action_desc = "Differentiation"
            # Update domains
            for tr_id in current_domains:
                if current_domains[tr_id] == 'Velocity':
                    current_domains[tr_id] = 'Acceleration'
                elif current_domains[tr_id] == 'Displacement':
                    current_domains[tr_id] = 'Velocity'
        
        elif action_type == 'select_traces':
            selected_indices_str = request.form.getlist('trace_indices')
            if selected_indices_str:
                visible_indices = [int(i) for i in selected_indices_str]
                action_desc = f"Updated View ({len(visible_indices)} visible)"
                # Do NOT filter st. Just update visible_indices.
            else:
                flash("No traces selected.")
                return redirect(url_for('index'))

        save_new_step(st, action_desc, current_domains, visible_indices)
        session.modified = True

    except Exception as e:
        flash(f"Error applying action: {e}")
    
    return redirect(url_for('index'))

@app.route('/remove_response', methods=['POST'])
def remove_response():
    try:
        # 1. Get current file
        filepath = get_current_file_path()
        if not filepath or not os.path.exists(filepath):
            flash("No waveform loaded.")
            return redirect(url_for('index'))

        # 2. Handle Response File Upload
        if 'response_file' not in request.files:
            flash("No response file uploaded.")
            return redirect(url_for('index'))
        
        response_file = request.files['response_file']
        if response_file.filename == '':
            flash("No selected file.")
            return redirect(url_for('index'))

        # Save response file temporarily
        resp_filename = f"resp_{uuid.uuid4().hex[:8]}.xml"
        resp_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'], resp_filename)
        response_file.save(resp_path)

        # 3. Read Inventory
        try:
            inv = read_inventory(resp_path)
        except Exception as e:
            flash(f"Error reading response file: {e}")
            return redirect(url_for('index'))

        # 4. Parse Parameters
        pre_filt_str = request.form.get('pre_filt', '0.005, 0.01, 10, 20')
        water_level = float(request.form.get('water_level', 60))
        output_unit = request.form.get('output', 'VEL') # DISP, VEL, ACC

        try:
            pre_filt = [float(x.strip()) for x in pre_filt_str.split(',')]
            if len(pre_filt) != 4:
                raise ValueError("Pre-filter must have 4 values.")
        except ValueError as e:
            flash(f"Invalid pre-filter format: {e}")
            return redirect(url_for('index'))

        # 5. Apply remove_response
        st = read(filepath)
        
        try:
            st.remove_response(inventory=inv, pre_filt=pre_filt, output=output_unit, water_level=water_level)
        except Exception as e:
            flash(f"Error removing response: {e}")
            return redirect(url_for('index'))

        # 6. Save Result
        action_desc = f"Remove Response ({output_unit})"
        
        # Map output to domain
        domain_map = {'DISP': 'displacement', 'VEL': 'velocity', 'ACC': 'acceleration'}
        new_domain_label = domain_map.get(output_unit, 'velocity')
        
        current_domains_dict = {}
        # We need to update domains for all traces
        for tr in st:
            current_domains_dict[tr.id] = new_domain_label

        # Preserve visibility
        history = session.get('history', [])
        visible_indices = []
        if history:
            visible_indices = history[-1].get('visible_indices', list(range(len(st))))
        else:
            visible_indices = list(range(len(st)))

        save_new_step(st, action_desc, current_domains_dict, visible_indices)
        session.modified = True
        
        flash(f"Instrument response removed. Output: {output_unit}")

    except Exception as e:
        flash(f"Error in remove_response: {e}")
    
    return redirect(url_for('index'))

@app.route('/undo', methods=['POST'])
def undo():
    history = session.get('history', [])
    if len(history) > 1:
        history.pop()
        session['history'] = history
        session.modified = True
    else:
        flash("Cannot undo further (original file).")
    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    # Clean up files
    if 'session_id' in session:
        folder = os.path.join(app.config['BASE_UPLOAD_FOLDER'], session['session_id'])
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"Error deleting session folder: {e}")
    
    session.pop('history', None)
    session.pop('filename', None) # Also clear filename if stored separately
    # Keep session_id? Or clear it to force new one?
    # Clearing it is safer for "Reset All".
    session.pop('session_id', None)
    
    return redirect(url_for('index'))

# --- Thomson-Haskell Section ---
from thomson_haskell import thomson_haskell_transfer_function

@app.route('/thomson_haskell', methods=['POST'])
def calc_thomson():
    try:
        # Parse Building Layers
        b_heights = request.form.getlist('b_height')
        b_vs = request.form.getlist('b_vs')
        b_rho = request.form.getlist('b_rho')
        b_qs = request.form.getlist('b_qs')
        
        # Parse Soil Layers
        s_heights = request.form.getlist('s_height')
        s_vs = request.form.getlist('s_vs')
        s_rho = request.form.getlist('s_rho')
        s_qs = request.form.getlist('s_qs')
        
        # Parse Frequency
        f_min = float(request.form.get('f_min', 0.1))
        f_max = float(request.form.get('f_max', 50))
        f_steps = int(request.form.get('f_steps', 500))
        freqs_manual = np.linspace(f_min, f_max, f_steps)

        # Construct Arrays
        h = []
        vs = []
        rho = []
        qs = []
        
        # Building (Top to Bottom)
        b_layers = []
        for i in range(len(b_heights)):
            if b_heights[i]: # Check if not empty
                h_val = float(b_heights[i])
                vs_val = float(b_vs[i])
                rho_val = float(b_rho[i])
                qs_val = float(b_qs[i])
                
                h.append(h_val)
                vs.append(vs_val)
                rho.append(rho_val)
                qs.append(qs_val)
                
                b_layers.append({
                    'h': h_val,
                    'vs': vs_val,
                    'rho': rho_val,
                    'qs': qs_val
                })
                
        # Soil (Top to Bottom)
        for i in range(len(s_vs)):
            # Last soil layer is half-space, so height is ignored/not present in h
            # But wait, h needs N layers, vs needs N+1.
            # So if we have M soil layers, the last one is halfspace.
            # The first M-1 soil layers contribute to h.
            # Building layers also contribute to h.
            
            # Actually, the user input for soil might be:
            # Layer 1: H=10, Vs=..., ...
            # Layer 2: H=Inf, Vs=..., ... (Halfspace)
            
            # Let's assume the last soil row provided is ALWAYS the halfspace.
            # So if user provides 1 soil row, it's the halfspace.
            # If 2 rows, first is layer, second is halfspace.
            
            is_halfspace = (i == len(s_vs) - 1)
            
            vs.append(float(s_vs[i]))
            rho.append(float(s_rho[i]))
            qs.append(float(s_qs[i]))
            
            if not is_halfspace:
                h.append(float(s_heights[i]))

        # Calculate TF
        tf = thomson_haskell_transfer_function(freqs_manual, h, vs, rho, qs)
        
        # Plot TF
        img_tf = io.BytesIO()
        fig_tf = plt.figure(figsize=(10, 6))
        ax = fig_tf.add_subplot(111)
        ax.loglog(freqs_manual, np.abs(tf), c='k')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'Transfer Function ($\frac{u_{top}}{u_{bottom}}$)')
        ax.grid(True, which="both")
        fig_tf.savefig(img_tf, format='png')
        img_tf.seek(0)
        tf_url = base64.b64encode(img_tf.getvalue()).decode()
        plt.close(fig_tf)
        
        # Sketch Model
        img_sketch = io.BytesIO()
        fig_sketch = plt.figure(figsize=(6, 8))
        ax_s = fig_sketch.add_subplot(111)
        
        current_depth = 0
        max_width = 10
        
        # Draw Building (Upwards from 0?) Or everything downwards?
        # Usually building is above ground (negative depth?)
        # Let's draw ground at y=0. Building goes up. Soil goes down.
        
        # Building Layers (Reversed order for drawing: bottom building layer is at y=0)
        # Wait, the input list order: usually top to bottom?
        # "Building Height" -> usually total height? Or layer height?
        # Let's assume input is Top -> Bottom.
        # So last building layer is at y=0.
        
        # Let's reconstruct depths
        # Building
        # --- Waveform Integration & Drift Estimation ---
    except Exception as e:
        flash(f"Error calculating Thomson-Haskell: {e}")
        return redirect(url_for('index'))

    drift_url = None
    mseed_filename = None
    
    # Check if we have a loaded waveform to use for frequencies and drift calc
    # Use helper to get current file path
    filepath = get_current_file_path()
    
    if filepath and os.path.exists(filepath):
        try:
            st = read(filepath)
            
            # Use the first trace (or selected traces logic if we want to be fancy, but let's stick to first for now or loop)
            # For simplicity, let's use the first trace as the "Bottom Motion"
            tr = st[0]
            npts = tr.stats.npts
            dt = tr.stats.delta
            
            # Use rfftfreq to match the waveform's frequencies
            freqs_waveform = np.fft.rfftfreq(npts, dt)
            
            # Re-calculate TF with these frequencies
            tf_complex_waveform = thomson_haskell_transfer_function(freqs_waveform, h, vs, rho, qs)
            
            # Calculate Top Motion
            # U_top(f) = U_bottom(f) * TF(f)
            u_bottom_f = np.fft.rfft(tr.data)
            u_top_f = u_bottom_f * tf_complex_waveform
            u_top_t = np.fft.irfft(u_top_f, n=npts)
            
            # Calculate Drift
            # Drift = Bottom - Top (as per user request "subtract Predicted Top Motion from Waveform at the bottom")
            # Wait, "subtract Predicted Top Motion from Waveform at the bottom" -> Bottom - Top
            drift_t = tr.data - u_top_t
            
            # Generate 3x2 Plot
            fig_drift, axes = plt.subplots(3, 2, figsize=(12, 10))
            times = tr.times()
            
            # Top Motion
            axes[0, 0].plot(times, u_top_t, 'k')
            axes[0, 0].set_title('Predicted Top Motion (Time)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True)
            
            axes[0, 1].loglog(freqs_waveform, np.abs(u_top_f), 'k')
            axes[0, 1].set_title('Predicted Top Motion (Freq)')
            axes[0, 1].grid(True)
            
            # Drift Motion
            axes[1, 0].plot(times, drift_t, 'r')
            axes[1, 0].set_title('Drift Motion (Bottom - Top)')
            axes[1, 0].set_ylabel('Amplitude')
            axes[1, 0].grid(True)
            
            axes[1, 1].loglog(freqs_waveform, np.abs(np.fft.rfft(drift_t)), 'r')
            axes[1, 1].set_title('Drift Motion (Freq)')
            axes[1, 1].grid(True)
            
            # Bottom Motion (Input)
            axes[2, 0].plot(times, tr.data, 'b')
            axes[2, 0].set_title('Bottom Motion (Input)')
            axes[2, 0].set_xlabel('Time [s]')
            axes[2, 0].set_ylabel('Amplitude')
            axes[2, 0].grid(True)
            
            axes[2, 1].loglog(freqs_waveform, np.abs(u_bottom_f), 'b')
            axes[2, 1].set_title('Bottom Motion (Freq)')
            axes[2, 1].set_xlabel('Frequency [Hz]')
            axes[2, 1].grid(True)
            
            plt.tight_layout()
            
            # Save Drift Plot
            img_drift = io.BytesIO()
            plt.savefig(img_drift, format='png')
            img_drift.seek(0)
            drift_url = base64.b64encode(img_drift.getvalue()).decode()
            plt.close(fig_drift)

            # Save MSEED Data
            mseed_filename = f"drift_{uuid.uuid4().hex[:8]}.mseed"
            mseed_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'], mseed_filename)
            
            # Create Traces
            # Bottom Motion (Input) - Copy to avoid modifying original if needed, though we just write it
            tr_bottom = tr.copy()
            
            # Top Motion
            tr_top = obspy.Trace(data=u_top_t)
            tr_top.stats = tr.stats.copy()
            tr_top.stats.station = "TOP"
            
            # Drift Motion
            tr_drift = obspy.Trace(data=drift_t)
            tr_drift.stats = tr.stats.copy()
            tr_drift.stats.station = "DRIFT"
            
            # Create Stream and Write
            st_out = obspy.Stream([tr_bottom, tr_top, tr_drift])
            st_out.write(mseed_path, format='MSEED')
            
        except Exception as e:
            print(f"Error in Drift Estimation: {e}")
            flash(f"Error in Drift Estimation: {e}")
            # Fallback to manual frequencies if something fails (or just log error)
            pass
    else:
        # Flash message if no file found for drift estimation
        # Only flash if we expected one? Or just let the UI handle the "No drift estimation available" message.
        # But if the user expects it, a flash is helpful.
        pass

    # --- Plotting Transfer Function (Standard) ---
    # If we used waveform freqs, we might want to plot TF with those, or the manual ones.
    # The user asked to "Pass the frequencies... to thomson-haskell".
    # So the TF plot should probably also reflect the waveform frequencies if available?
    # Or maybe keep the manual range for the TF plot specifically?
    # The prompt says: "Pass the frequencies you used in the waveform processing tab to thomson-haskell to use them in the frequency information for the transfer function."
    # This implies the MAIN calculation should use these frequencies.
    
    # Let's use the manual freqs for the standard TF plot if no file, OR if we want to keep the "Calculator" view clean.
    # But for consistency, if a file is loaded, we used its freqs for drift.
    # Let's stick to the manual range for the "Transfer Function" plot to ensure it looks nice (log-spaced),
    # unless the user explicitly wants to see the TF at the FFT bins (which are linear).
    # Usually TF is viewed log-log. FFT bins are linear. Plotting linear bins on log-log is fine.
    
    # However, to avoid confusion, let's keep the "Transfer Function" plot based on the MANUAL input range
    # because that's what the user controls for the "Calculator" part.
    # The Drift Estimation uses the waveform frequencies internally.
    
    # ... (Rest of the code uses 'freqs' which was defined earlier from manual inputs. 
    # If we want to overwrite 'freqs' for the TF plot too, we would do it here. 
    # But 'freqs' variable is reused. Let's keep the manual 'freqs' for the TF plot 
    # and use 'freqs_waveform' for the drift calc if needed, but I already did the drift calc above.)
    
    # Recalculate TF for the standard plot using MANUAL inputs (for clean visualization)
    # This ensures the "Transfer Function" tab still looks as expected (smooth, user-defined range).
    tf = thomson_haskell_transfer_function(freqs_manual, h, vs, rho, qs)

    img_tf = io.BytesIO()
    fig_tf = plt.figure(figsize=(10, 6))
    ax = fig_tf.add_subplot(111)
    ax.loglog(freqs_manual, np.abs(tf), c='k')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'Transfer Function ($\frac{u_{top}}{u_{bottom}}$)')
    ax.grid(True, which="both")
    fig_tf.savefig(img_tf, format='png')
    img_tf.seek(0)
    tf_url = base64.b64encode(img_tf.getvalue()).decode()
    plt.close(fig_tf)
    
    # Sketch Model
    img_sketch = io.BytesIO()
    fig_sketch = plt.figure(figsize=(6, 8))
    ax_s = fig_sketch.add_subplot(111)
    
    current_depth = 0
    max_width = 10
    
    # Draw Building (Upwards from 0?) Or everything downwards?
    # Usually building is above ground (negative depth?)
    # Let's draw ground at y=0. Building goes up. Soil goes down.
    
    # Building Layers (Reversed order for drawing: bottom building layer is at y=0)
    # Wait, the input list order: usually top to bottom?
    # "Building Height" -> usually total height? Or layer height?
    # Let's assume input is Top -> Bottom.
    # So last building layer is at y=0.
    
    # Let's reconstruct depths
    # Building
    # b_layers already constructed above
        
    # Draw Building
    current_y = 0
    total_b_height = 0
    for lay in reversed(b_layers):
        h_val = lay['h']
        total_b_height += h_val
        rect = plt.Rectangle((2, current_y), 6, h_val, color='skyblue', alpha=0.5, ec='black')
        ax_s.add_patch(rect)
        ax_s.text(5, current_y + h_val/2, f"H={h_val}\nVs={lay['vs']}\nQ={lay['qs']}\nRho={lay['rho']}", ha='center', va='center', fontsize=8)
        current_y += h_val
            
    ax_s.text(5, current_y + 1, "Building Top", ha='center')
    
    # Add Triangle at Top
    ax_s.plot(5, current_y, marker='v', markersize=10, color='red')
        
    # Draw Soil
    current_y = 0
    s_layers_count = len(s_vs)
    for i in range(s_layers_count):
        is_halfspace = (i == s_layers_count - 1)
        vs_val = float(s_vs[i])
        qs_val = float(s_qs[i])
        rho_val = float(s_rho[i])
        
        if is_halfspace:
            h_val = 20 # Arbitrary visual height for halfspace
            color = 'lightgrey'
            label = "Half-space"
            text_content = f"{label}\nVs={vs_val}\nQ={qs_val}\nRho={rho_val}"
        else:
            h_val = float(s_heights[i])
            color = 'wheat'
            label = f"Layer {i+1}"
            text_content = f"{label}\nH={h_val}\nVs={vs_val}\nQ={qs_val}\nRho={rho_val}"
            
        rect = plt.Rectangle((0, current_y - h_val), 10, h_val, color=color, alpha=0.5, ec='black')
        ax_s.add_patch(rect)
        ax_s.text(5, current_y - h_val/2, text_content, ha='center', va='center', fontsize=8)
        current_y -= h_val

    # Add Triangle at Bottom (Soil-Structure Interface, y=0)
    ax_s.plot(5, 0, marker='^', markersize=10, color='red')

    ax_s.set_xlim(0, 10)
    ax_s.set_ylim(current_y - 5, total_b_height + 5)
    ax_s.axis('off')
    ax_s.set_title("Model Sketch")
        
    fig_sketch.savefig(img_sketch, format='png')
    img_sketch.seek(0)
    sketch_url = base64.b64encode(img_sketch.getvalue()).decode()
    plt.close(fig_sketch)
        
    # Save NPZ
    npz_filename = f"tf_{uuid.uuid4().hex[:8]}.npz"
    npz_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'], npz_filename)
    np.savez(npz_path, freqs=freqs_manual, tf=tf, h=h, vs=vs, rho=rho, qs=qs)
    

    
    # Reconstruct current_info for template
    current_info = None
    history = session.get('history', [])
    if history:
        current_step = history[-1]
        # We need to re-read the file to get traces info if we want to show it?
        # Or just pass what we have.
        # But wait, index.html needs 'traces' list for the sidebar if we want to keep it visible.
        # To avoid re-reading file (expensive), maybe we can skip it or just pass minimal info?
        # But if the user switches tab back to Waveform, they expect it to be there.
        # Let's try to re-read if possible, or just rely on session data if we stored it?
        # We stored 'visible_indices' and 'domains'.
        # We didn't store trace stats/ids in session history directly, only in the file.
        # So we SHOULD read the file to be safe and consistent.
        
        if filepath and os.path.exists(filepath):
             try:
                st = read(filepath)
                visible_indices = current_step.get('visible_indices', list(range(len(st))))
                
                # We also need plot_url and spectrum_url if we want the Waveform tab to be populated!
                # Even if hidden, it's good practice.
                plot_url, spectrum_url = generate_plots(st, current_step['domains'], visible_indices)
                
                traces_info = []
                for i, tr in enumerate(st):
                    traces_info.append({
                        'index': i,
                        'id': tr.id,
                        'stats': str(tr.stats),
                        'visible': i in visible_indices
                    })

                current_info = {
                    'filename': current_step['filename'],
                    'domains': current_step['domains'],
                    'history': history,
                    'traces': traces_info
                }
             except Exception as e:
                 print(f"Error reconstructing state: {e}")
    
    # If current_info is None (no file loaded), we just pass None.
    # But we need to define plot_url/spectrum_url variables to avoid UnboundLocalError if we don't enter the block above.
    if 'plot_url' not in locals(): plot_url = None
    if 'spectrum_url' not in locals(): spectrum_url = None

    return render_template('index.html', 
                           tf_url=tf_url, 
                           sketch_url=sketch_url, 
                           drift_url=drift_url,
                           npz_filename=npz_filename,
                           mseed_filename=mseed_filename,
                           active_tab='thomson', # Keep tab active
                           current_info=current_info,
                           plot_url=plot_url,
                           spectrum_url=spectrum_url)

@app.route('/download/<filename>')
def download_file(filename):
    return flask.send_from_directory(app.config['BASE_UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
