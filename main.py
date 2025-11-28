# =========================================================
# CHRONO-INSIGHT DASHBOARD — Refined Professional Version
# =========================================================
import os
import shutil
import uuid
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context, no_update
from wordcloud import WordCloud
import base64
import io

# Import the processing functions from our NLP pipeline
try:
    from nlp_pipeline import process_pdfs, create_nlp_enhanced_dataframes
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("NLP pipeline not available - using demo mode only")

# ---------- Premium Color Palette ----------
COLORS = {
    'sidebar_dark': '#0A1828',        # Dark blue for sidebar
    'sidebar_accent': '#178582',       # Turquoise for sidebar accents
    'gold': '#BFA181',                 # Gold for text on dark
    'main_bg': '#F5F1E8',             # Cream beige for main section
    'card_bg': '#FFFFFF',              # White for cards
    'text_dark': '#2C3E50',            # Dark text for main area
    'text_light': '#7F8C8D',           # Light text
    'accent_light': '#3498DB',         # Light blue for accents
    'success': '#27AE60',              # Green for success states
    'warning': '#E67E22',              # Orange for warnings
    'white': '#FFFFFF'
}

# ---------- Configuration ----------
BASE_PATH = "/project/workspace/"
DEFAULT_DATA_PATH = f"{BASE_PATH}/default_data"
USER_SESSIONS_PATH = f"{BASE_PATH}/user_sessions"
TEMP_PATH = f"{BASE_PATH}/temp"

# Create directories if they don't exist
for path in [DEFAULT_DATA_PATH, USER_SESSIONS_PATH, TEMP_PATH]:
    os.makedirs(path, exist_ok=True)

# ---------- Session Manager Class ----------
class SessionManager:
    def __init__(self):
        self.current_mode = "default"  # "default" or "user"
        self.current_session_id = None
        self.sessions = {}
        # Add tracking for unprocessed files
        self.has_unprocessed_files = False
        
    def create_new_session(self):
        """Create a new session for user uploads"""
        session_id = str(uuid.uuid4())[:8]
        session_path = os.path.join(USER_SESSIONS_PATH, f"session_{session_id}")
        upload_path = os.path.join(session_path, "uploaded_pdfs")
        processed_path = os.path.join(session_path, "processed_data")
        
        # Create session directory structure
        os.makedirs(upload_path, exist_ok=True)
        os.makedirs(processed_path, exist_ok=True)
        
        # Initialize session info
        self.sessions[session_id] = {
            'created_at': datetime.now().isoformat(),
            'uploaded_files': [],
            'processed_files': [],
            'session_path': session_path,
            'upload_path': upload_path,
            'processed_path': processed_path
        }
        
        self.current_session_id = session_id
        self.current_mode = "user"
        self.has_unprocessed_files = True  # New session has files to process
        print(f"DEBUG: Created new session {session_id} at {session_path}")
        return session_id
    
    def add_files_to_session(self, session_id, file_contents, filenames):
        """Add PDF files to an existing session"""
        if session_id not in self.sessions:
            return False, "Session not found"
            
        session = self.sessions[session_id]
        saved_files = []
        
        for content, filename in zip(file_contents, filenames):
            if not filename.lower().endswith('.pdf'):
                continue
                
            try:
                # Decode the base64 data URL
                if content.startswith('data:application/pdf;base64,'):
                    # Extract the base64 part after the comma
                    base64_data = content.split(',')[1]
                    file_data = base64.b64decode(base64_data)
                else:
                    # If it's already raw base64 (shouldn't happen but just in case)
                    file_data = base64.b64decode(content)
                
                # Save file to session upload directory
                file_path = os.path.join(session['upload_path'], filename)
                with open(file_path, 'wb') as f:
                    f.write(file_data)  # Now writing bytes, not string
                
                saved_files.append(filename)
                print(f"DEBUG: Saved file {filename} to {file_path}")
                
            except Exception as e:
                print(f"DEBUG: Error saving file {filename}: {e}")
                continue
        
        # Update session with new files
        session['uploaded_files'].extend(saved_files)
        session['last_updated'] = datetime.now().isoformat()
        
        # MARK: New files added, enable processing
        self.has_unprocessed_files = True
        
        return True, f"Added {len(saved_files)} files to session"
    
    def get_session_files(self, session_id):
        """Get list of PDF files in a session"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]['uploaded_files']
    
    def process_session_pdfs(self, session_id):
        """Process all PDFs in a session using the NLP pipeline"""
        if not NLP_AVAILABLE:
            return False, "NLP processing not available"
            
        if session_id not in self.sessions:
            return False, "Session not found"
            
        session = self.sessions[session_id]
        upload_path = session['upload_path']
        processed_path = session['processed_path']
        
        print(f"DEBUG: Processing PDFs from {upload_path}")
        print(f"DEBUG: Will save to {processed_path}")
        
        try:
            # Process PDFs using the NLP pipeline
            all_insights = process_pdfs(upload_path)
            
            if not all_insights:
                print("DEBUG: No insights extracted from PDFs")
                return False, "No insights extracted from PDFs"
            
            print(f"DEBUG: Extracted {len(all_insights)} insights")
            
            # Create dataframes and save to session processed data
            insights_df, domain_df, temporal_df, pattern_df, entity_df = create_nlp_enhanced_dataframes(all_insights)
            
            print(f"DEBUG: Created dataframes - Insights: {len(insights_df)}, Domain: {len(domain_df)}")
            
            # Save processed data to session folder
            insights_df.to_csv(os.path.join(processed_path, 'nlp_ai_insights.csv'), index=False)
            domain_df.to_csv(os.path.join(processed_path, 'nlp_domain_analysis.csv'), index=False)
            temporal_df.to_csv(os.path.join(processed_path, 'nlp_temporal_analysis.csv'), index=False)
            pattern_df.to_csv(os.path.join(processed_path, 'nlp_pattern_analysis.csv'), index=False)
            entity_df.to_csv(os.path.join(processed_path, 'nlp_entity_analysis.csv'), index=False)
            
            # VERIFY FILES WERE CREATED
            created_files = []
            for filename in ['nlp_ai_insights.csv', 'nlp_domain_analysis.csv', 'nlp_temporal_analysis.csv', 'nlp_pattern_analysis.csv', 'nlp_entity_analysis.csv']:
                file_path = os.path.join(processed_path, filename)
                if os.path.exists(file_path):
                    created_files.append(filename)
                    print(f"DEBUG: Successfully created {file_path}")
                    # Check file size
                    file_size = os.path.getsize(file_path)
                    print(f"DEBUG: File {filename} size: {file_size} bytes")
                else:
                    print(f"DEBUG: FAILED to create {file_path}")
            
            session['processed_files'] = created_files
            session['last_processed'] = datetime.now().isoformat()
            session['insight_count'] = len(all_insights)
            
            print(f"DEBUG: Session updated with {len(created_files)} processed files")
            
            # MARK: Processing complete, reset unprocessed files flag
            self.has_unprocessed_files = False
            
            return True, f"Successfully processed {len(all_insights)} insights"
            
        except Exception as e:
            print(f"DEBUG: Error in process_session_pdfs: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, f"Error processing PDFs: {str(e)}"
    
    def clear_session(self, session_id):
        """Clear a session and all its files"""
        if session_id in self.sessions:
            session_path = self.sessions[session_id]['session_path']
            try:
                shutil.rmtree(session_path)
                print(f"DEBUG: Cleared session {session_id}")
            except Exception as e:
                print(f"DEBUG: Error clearing session {session_id}: {e}")
            del self.sessions[session_id]
            
        if self.current_session_id == session_id:
            self.current_session_id = None
            self.current_mode = "default"
            self.has_unprocessed_files = False

    def switch_to_default(self):
        """Switch back to default data"""
        self.current_mode = "default"
        self.current_session_id = None
        self.has_unprocessed_files = False
        print("DEBUG: Switched to default data mode")

    def get_current_data_path(self):
        """Get the path to current data based on mode"""
        if self.current_mode == "default":
            return DEFAULT_DATA_PATH
        elif self.current_mode == "user" and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]['processed_path']
        else:
            return DEFAULT_DATA_PATH

    def get_processing_status(self, session_id):
        """Get current processing status for display"""
        if session_id not in self.sessions:
            return "No active session"
        
        session = self.sessions[session_id]
        file_count = len(session.get('uploaded_files', []))
        processed_count = session.get('insight_count', 0)
        
        if self.has_unprocessed_files:
            return f"Ready to process {file_count} PDFs"
        elif processed_count > 0:
            return f"Processed {file_count} PDFs ({processed_count} insights)"
        else:
            return f"{file_count} PDFs uploaded - Ready to process"

# ---------- Initialize Session Manager ----------
session_manager = SessionManager()

# ---------- Data Loading Helper ----------
def load_current_data():
    """Load data based on current session mode with retry logic"""
    data_path = session_manager.get_current_data_path()
    
    print(f"DEBUG: Loading data from {data_path}")
    print(f"DEBUG: Current mode: {session_manager.current_mode}")
    print(f"DEBUG: Session ID: {session_manager.current_session_id}")
    
    # List files in the directory for debugging
    try:
        if os.path.exists(data_path):
            files = os.listdir(data_path)
            print(f"DEBUG: Files in {data_path}: {files}")
        else:
            print(f"DEBUG: Path does not exist: {data_path}")
    except Exception as e:
        print(f"DEBUG: Error listing directory: {e}")

    dataframes = {}
    file_map = {
        'ai': 'nlp_ai_insights.csv',
        'domain': 'nlp_domain_analysis.csv', 
        'temporal': 'nlp_temporal_analysis.csv',
        'pattern': 'nlp_pattern_analysis.csv',
        'entity': 'nlp_entity_analysis.csv'
    }
    
    for key, filename in file_map.items():
        file_path = os.path.join(data_path, filename)
        print(f"DEBUG: Looking for {file_path}")
        
        if os.path.exists(file_path):
            try:
                dataframes[key] = pd.read_csv(file_path)
                print(f"DEBUG: Successfully loaded {filename} with {len(dataframes[key])} rows")
            except Exception as e:
                print(f"DEBUG: Error reading {filename}: {e}")
                dataframes[key] = pd.DataFrame()
        else:
            print(f"DEBUG: File not found: {file_path}")
            dataframes[key] = pd.DataFrame()
    
    return (dataframes.get('ai', pd.DataFrame()), 
            dataframes.get('domain', pd.DataFrame()),
            dataframes.get('temporal', pd.DataFrame()),
            dataframes.get('pattern', pd.DataFrame()),
            dataframes.get('entity', pd.DataFrame()))

# ---------- Helper: WordCloud Image ----------
def make_wordcloud(data, column):
    if data.empty or column not in data.columns:
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    text = " ".join(data[column].dropna().astype(str))
    if not text.strip():
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
    wc = WordCloud(width=800, height=400, background_color=COLORS['main_bg']).generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# ---------- Custom Styled Components ----------
def create_metric_card(title, value, subtitle="", color=COLORS['sidebar_accent']):
    """Create a beautiful metric card"""
    return html.Div([
        html.Div([
            html.H3(value, style={'margin': '0', 'color': COLORS['text_dark'], 'fontSize': '32px', 'fontWeight': '700', 'lineHeight': '1.2'}),
            html.P(title, style={'margin': '5px 0 0 0', 'color': COLORS['text_light'], 'fontSize': '14px', 'fontWeight': '600'}),
            html.P(subtitle, style={'margin': '2px 0 0 0', 'color': COLORS['accent_light'], 'fontSize': '12px', 'opacity': '0.8'}) if subtitle else None
        ], style={
            'padding': '25px',
            'background': COLORS['card_bg'],
            'borderRadius': '15px',
            'boxShadow': '0 4px 20px rgba(0,0,0,0.08)',
            'textAlign': 'center',
            'height': '100%',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center',
            'border': f'1px solid {COLORS["main_bg"]}',
            'transition': 'all 0.3s ease'
        })
    ], style={'height': '100%'})

def create_section_header(title, description):
    """Create a stylish section header"""
    return html.Div([
        html.H2(title, style={
            'color': COLORS['text_dark'],
            'marginBottom': '8px',
            'fontSize': '28px',
            'fontWeight': '700'
        }),
        html.P(description, style={
            'color': COLORS['text_light'],
            'marginBottom': '30px',
            'fontSize': '16px',
        })
    ])

# ---------- Premium App Layout ----------
app = Dash(__name__, title="Chrono-Insight", suppress_callback_exceptions=True)

# Enhanced Sidebar with premium styling
sidebar = html.Div([
    # Header with logo
    html.Div([
        html.Div([
            html.H1("Chrono-Insight", style={
                'color': COLORS['gold'], 
                'marginBottom': '5px',
                'fontSize': '24px',
                'fontWeight': '700',
                'letterSpacing': '0.5px'
            }),
            html.P("Temporal AI Analysis Platform", style={
                'color': COLORS['gold'],
                'fontSize': '12px',
                'opacity': '0.8',
                'marginBottom': '20px',
                'letterSpacing': '1px'
            })
        ], style={'textAlign': 'center', 'padding': '20px 0', 'borderBottom': f'1px solid {COLORS["sidebar_accent"]}30'})
    ]),
    
    # Data Source Section
    html.Div([
        html.H5("DATA SOURCE", style={
            'color': COLORS['gold'], 
            'marginBottom': '15px',
            'fontSize': '11px',
            'letterSpacing': '2px',
            'opacity': '0.7',
            'textTransform': 'uppercase'
        }),
        dcc.RadioItems(
            id='data-source-toggle',
            options=[
                {'label': ' Sample Data', 'value': 'default'},
                {'label': ' Your Analysis', 'value': 'user'}
            ],
            value='default',
            labelStyle={
                'display': 'flex', 
                'alignItems': 'center',
                'marginBottom': '12px',
                'padding': '12px 15px',
                'backgroundColor': f'{COLORS["sidebar_accent"]}15',
                'borderRadius': '12px',
                'cursor': 'pointer',
                'transition': 'all 0.3s ease',
                'border': f'1px solid {COLORS["sidebar_accent"]}30'
            },
            inputStyle={'marginRight': '10px'}
        ),
    ], style={'marginBottom': '25px', 'padding': '0 10px'}),
    
    html.Hr(style={'borderColor': f'{COLORS["sidebar_accent"]}30', 'margin': '20px 0'}),
    
    # Upload Section
    html.Div([
        html.H5("PDF ANALYSIS", style={
            'color': COLORS['gold'],
            'marginBottom': '15px',
            'fontSize': '11px',
            'letterSpacing': '2px',
            'opacity': '0.7',
            'textTransform': 'uppercase'
        }),
        
        # Enhanced Upload Component
        dcc.Upload(
            id='upload-pdfs',
            children=html.Div([
                html.Div([
                    html.Div("+", style={'fontSize': '32px', 'color': COLORS['sidebar_accent'], 'marginBottom': '10px'}),
                    html.Div([
                        'Drag & Drop or ',
                        html.A('Select PDFs', style={'color': COLORS['gold'], 'textDecoration': 'underline', 'fontWeight': '600'})
                    ], style={'fontSize': '14px', 'color': COLORS['gold']})
                ], style={'textAlign': 'center'})
            ]),
            style={
                'width': '100%', 
                'height': '140px', 
                'lineHeight': 'normal',
                'border': f'2px dashed {COLORS["sidebar_accent"]}',
                'borderRadius': '15px',
                'textAlign': 'center',
                'marginBottom': '15px', 
                'cursor': 'pointer',
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'center',
                'alignItems': 'center',
                'transition': 'all 0.3s ease',
                'backgroundColor': f'{COLORS["sidebar_accent"]}08'
            },
            multiple=True
        ),
        
        # Upload Options
        html.Div([
            html.Label("Add to:", style={'color': COLORS['gold'], 'marginBottom': '8px', 'fontSize': '13px'}),
            dcc.RadioItems(
                id='upload-mode',
                options=[
                    {'label': ' New Analysis', 'value': 'new'},
                    {'label': ' Current Analysis', 'value': 'add'}
                ],
                value='new',
                labelStyle={
                    'display': 'flex', 
                    'alignItems': 'center',
                    'fontSize': '12px',
                    'marginBottom': '8px',
                    'padding': '10px 12px',
                    'backgroundColor': f'{COLORS["sidebar_accent"]}15',
                    'borderRadius': '10px',
                    'cursor': 'pointer',
                    'border': f'1px solid {COLORS["sidebar_accent"]}30'
                }
            ),
        ], style={'marginBottom': '15px', 'padding': '0 5px'}),
        
        # Enhanced Process Button
        html.Button(
            'Process PDFs', 
            id='process-button',
            n_clicks=0,
            style={
                'width': '100%', 
                'padding': '16px', 
                'marginBottom': '15px',
                'backgroundColor': COLORS['sidebar_accent'],
                'color': COLORS['sidebar_dark'],
                'border': 'none',
                'borderRadius': '12px',
                'cursor': 'pointer',
                'fontSize': '14px',
                'fontWeight': '700',
                'transition': 'all 0.3s ease',
                'letterSpacing': '0.5px'
            },
            disabled=not NLP_AVAILABLE
        ),
        
        # Enhanced Session Info
        html.Div(id='session-info', style={
            'backgroundColor': f'{COLORS["sidebar_accent"]}15',
            'padding': '20px',
            'borderRadius': '15px',
            'marginBottom': '15px',
            'border': f'1px solid {COLORS["sidebar_accent"]}30'
        }),
        
        # Enhanced Reset Button
        html.Button(
            'Reset to Sample Data', 
            id='reset-default',
            n_clicks=0,
            style={
                'width': '100%', 
                'padding': '14px', 
                'marginBottom': '15px',
                'backgroundColor': 'transparent',
                'color': COLORS['gold'],
                'border': f'2px solid {COLORS["gold"]}',
                'borderRadius': '12px',
                'cursor': 'pointer',
                'fontSize': '13px',
                'fontWeight': '600',
                'transition': 'all 0.3s ease'
            }
        ),
        
    ], id='upload-section', style={'padding': '0 10px'}),
    
    html.Hr(style={'borderColor': f'{COLORS["sidebar_accent"]}30', 'margin': '20px 0'}),
    
    # Current Status
    html.Div(id='current-status', style={
        'color': COLORS['gold'], 
        'fontSize': '12px', 
        'textAlign': 'center',
        'opacity': '0.7',
        'padding': '0 10px'
    }),
    
], style={
    'width': '300px', 
    'padding': '0', 
    'backgroundColor': COLORS['sidebar_dark'],
    'height': '100vh', 
    'position': 'fixed', 
    'left': 0, 
    'top': 0,
    'overflowY': 'auto',
    'background': f'linear-gradient(180deg, {COLORS["sidebar_dark"]} 0%, #0d1f35 100%)'
})

# Premium Main Content Area
main_content = html.Div([
    # Hero Header
    html.Div([
        html.Div([
            html.H1("Chrono-Insight Analytics", style={
                'textAlign': 'center', 
                'marginBottom': '15px',
                'color': COLORS['text_dark'],
                'fontSize': '42px',
                'fontWeight': '800',
                'background': f'linear-gradient(135deg, {COLORS["text_dark"]}, {COLORS["accent_light"]})',
                'backgroundClip': 'text',
                'WebkitBackgroundClip': 'text',
                'color': 'transparent',
            }),
            html.H4("Advanced Temporal Analysis of AI Impact Across Domains", 
                   style={
                       'textAlign': 'center', 
                       'color': COLORS['text_light'], 
                       'marginBottom': '40px',
                       'fontSize': '18px',
                       'fontWeight': '400',
                   }),
        ], style={'padding': '50px 0 30px 0'})
    ]),
    
    # Data Source Indicator
    html.Div(id='data-source-indicator', style={'marginBottom': '30px'}),
    
    # Metrics Dashboard
    html.Div(id='metrics-dashboard', style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))',
        'gap': '20px',
        'marginBottom': '40px'
    }),
    
    # Hidden refresh trigger
    dcc.Store(id='data-refresh-trigger'),
    
    # Enhanced Tabs with curved design
    html.Div([
        dcc.Tabs(
            id="tabs", 
            value="ai",
            children=[
                dcc.Tab(
                    label="Sentence Explorer", 
                    value="ai",
                    style={
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['text_light'],
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid transparent'
                    },
                    selected_style={
                        'backgroundColor': COLORS['card_bg'],
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['sidebar_accent'],
                        'fontWeight': '700',
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid {COLORS["sidebar_accent"]}',
                        'boxShadow': '0 -4px 20px rgba(0,0,0,0.1)'
                    }
                ),
                dcc.Tab(
                    label="Domain Analysis", 
                    value="domain",
                    style={
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['text_light'],
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid transparent'
                    },
                    selected_style={
                        'backgroundColor': COLORS['card_bg'],
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['sidebar_accent'],
                        'fontWeight': '700',
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid {COLORS["sidebar_accent"]}',
                        'boxShadow': '0 -4px 20px rgba(0,0,0,0.1)'
                    }
                ),
                dcc.Tab(
                    label="Temporal Trends", 
                    value="temporal",
                    style={
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['text_light'],
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid transparent'
                    },
                    selected_style={
                        'backgroundColor': COLORS['card_bg'],
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['sidebar_accent'],
                        'fontWeight': '700',
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid {COLORS["sidebar_accent"]}',
                        'boxShadow': '0 -4px 20px rgba(0,0,0,0.1)'
                    }
                ),
                dcc.Tab(
                    label="Pattern Insights", 
                    value="pattern",
                    style={
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['text_light'],
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid transparent'
                    },
                    selected_style={
                        'backgroundColor': COLORS['card_bg'],
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['sidebar_accent'],
                        'fontWeight': '700',
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid {COLORS["sidebar_accent"]}',
                        'boxShadow': '0 -4px 20px rgba(0,0,0,0.1)'
                    }
                ),
                dcc.Tab(
                    label="Entity Analysis", 
                    value="entity",
                    style={
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['text_light'],
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid transparent'
                    },
                    selected_style={
                        'backgroundColor': COLORS['card_bg'],
                        'border': 'none',
                        'padding': '15px 25px',
                        'color': COLORS['sidebar_accent'],
                        'fontWeight': '700',
                        'borderRadius': '25px 25px 0 0',
                        'marginRight': '8px',
                        'borderBottom': f'3px solid {COLORS["sidebar_accent"]}',
                        'boxShadow': '0 -4px 20px rgba(0,0,0,0.1)'
                    }
                ),
            ]
        )
    ], style={
        'marginBottom': '0px',
        'backgroundColor': 'transparent',
        'borderBottom': f'1px solid {COLORS["main_bg"]}'
    }),
    
    # Tab Content
    html.Div(id="content", style={
        "marginTop": "0px",
        "padding": "30px 0",
        "minHeight": "600px"
    })
], style={
    'marginLeft': '300px', 
    'padding': '0 40px',
    'minHeight': '100vh',
    'background': COLORS['main_bg'],
    'color': COLORS['text_dark']
})

app.layout = html.Div([sidebar, main_content])

# ---------- Enhanced Callbacks ----------

@app.callback(
    Output('data-source-indicator', 'children'),
    Output('current-status', 'children'),
    Output('metrics-dashboard', 'children'),
    Input('data-source-toggle', 'value'),
    Input('reset-default', 'n_clicks'),
    Input('data-refresh-trigger', 'data')
)
def update_data_source_indicator(data_source, reset_clicks, refresh_data):
    """Update the data source indicator, status, and metrics"""
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-default.n_clicks':
        session_manager.switch_to_default()
        print("DEBUG: Reset to default data triggered")
    
    session_manager.current_mode = data_source
    
    # Load data for metrics
    ai, domain, temporal, pattern, entity = load_current_data()
    
    # Create metrics cards
    if data_source == 'default':
        indicator = html.Div([
            html.Span("USING SAMPLE DATA", 
                     style={
                         'backgroundColor': f'{COLORS["accent_light"]}15', 
                         'padding': '12px 30px', 
                         'borderRadius': '20px', 
                         'color': COLORS['accent_light'], 
                         'fontWeight': '700',
                         'border': f'2px solid {COLORS["accent_light"]}30',
                         'fontSize': '14px',
                         'letterSpacing': '1px'
                     })
        ])
        status = "Viewing pre-analyzed sample data"
        
        # Sample metrics
        metrics = [
            create_metric_card("Total Insights", str(len(ai)), "across all domains"),
            create_metric_card("Domains Covered", str(len(domain['domain'].unique()) if not domain.empty else 0), "different categories"),
            create_metric_card("Time Periods", str(len(temporal['timeframe'].unique()) if not temporal.empty else 0), "analyzed"),
            create_metric_card("Entities Found", str(len(entity)) if not entity.empty else "0", "key entities identified")
        ]
        
    else:
        if session_manager.current_session_id:
            session = session_manager.sessions.get(session_manager.current_session_id, {})
            file_count = len(session.get('uploaded_files', []))
            insight_count = session.get('insight_count', 0)
            status = f"Active analysis: {file_count} PDFs, {insight_count} insights"
            
            # User session metrics
            metrics = [
                create_metric_card("PDFs Uploaded", str(file_count), "documents processed"),
                create_metric_card("Insights Found", str(insight_count), "extracted insights"),
                create_metric_card("Domains", str(len(domain['domain'].unique()) if not domain.empty else 0), "categories identified"),
                create_metric_card("Status", "Complete" if insight_count > 0 else "Ready", "analysis status")
            ]
        else:
            status = "Ready for PDF upload and analysis"
            metrics = [
                create_metric_card("PDFs Uploaded", "0", "awaiting documents"),
                create_metric_card("Insights Found", "0", "start analysis"),
                create_metric_card("Analysis", "Ready", "for processing"),
                create_metric_card("Status", "Waiting", "for upload")
            ]
        
        indicator = html.Div([
            html.Span("YOUR ANALYSIS", 
                     style={
                         'backgroundColor': f'{COLORS["success"]}15', 
                         'padding': '12px 30px', 
                         'borderRadius': '20px', 
                         'color': COLORS['success'], 
                         'fontWeight': '700',
                         'border': f'2px solid {COLORS["success"]}30',
                         'fontSize': '14px',
                         'letterSpacing': '1px'
                     })
        ])
    
    return indicator, status, metrics

@app.callback(
    Output('session-info', 'children'),
    Input('data-source-toggle', 'value'),
    Input('upload-pdfs', 'contents'),
    Input('process-button', 'n_clicks'),
    Input('data-refresh-trigger', 'data')
)
def update_session_info(data_source, upload_contents, process_clicks, refresh_data):
    """Update session information display with processing status"""
    if data_source != 'user' or not session_manager.current_session_id:
        return html.Div("No active analysis session", style={'color': COLORS['gold'], 'textAlign': 'center', 'fontSize': '14px'})
    
    session = session_manager.sessions.get(session_manager.current_session_id, {})
    file_count = len(session.get('uploaded_files', []))
    insight_count = session.get('insight_count', 0)
    status = session_manager.get_processing_status(session_manager.current_session_id)
    
    return html.Div([
        html.H5("SESSION INFO", style={
            'color': COLORS['gold'], 
            'marginBottom': '15px',
            'fontSize': '12px',
            'letterSpacing': '2px',
            'borderBottom': f'1px solid {COLORS["sidebar_accent"]}30',
            'paddingBottom': '10px',
            'textTransform': 'uppercase'
        }),
        html.Div([
            html.Span("Documents: ", style={'color': COLORS['gold'], 'opacity': '0.8'}),
            html.Span(f"{file_count}", style={'color': COLORS['gold'], 'fontWeight': '700'})
        ], style={'marginBottom': '10px', 'fontSize': '14px'}),
        html.Div([
            html.Span("Insights: ", style={'color': COLORS['gold'], 'opacity': '0.8'}),
            html.Span(f"{insight_count}", style={'color': COLORS['gold'], 'fontWeight': '700'})
        ], style={'marginBottom': '10px', 'fontSize': '14px'}),
        html.Div([
            html.Span("Created: ", style={'color': COLORS['gold'], 'opacity': '0.8'}),
            html.Span(f"{session.get('created_at', '')[:16]}", style={'color': COLORS['gold'], 'fontSize': '12px'})
        ], style={'marginBottom': '15px', 'fontSize': '12px'}),
        html.Hr(style={'borderColor': f'{COLORS["sidebar_accent"]}30', 'margin': '15px 0'}),
        html.Div(status, style={
            'color': COLORS['sidebar_accent'], 
            'fontWeight': '600', 
            'textAlign': 'center',
            'padding': '12px',
            'backgroundColor': f'{COLORS["sidebar_accent"]}20',
            'borderRadius': '10px',
            'fontSize': '13px',
            'border': f'1px solid {COLORS["sidebar_accent"]}30'
        })
    ])

@app.callback(
    Output('process-button', 'children'),
    Output('process-button', 'disabled'),
    Input('upload-pdfs', 'contents'),
    Input('data-source-toggle', 'value'),
    State('process-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_process_button_state(upload_contents, data_source, process_clicks):
    """Update button state based on whether there are unprocessed files"""
    ctx = callback_context
    
    # If files were just uploaded and we're in user mode, enable the button
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'upload-pdfs.contents':
        if upload_contents and data_source == 'user':
            return 'Process PDFs', False
    
    # If we're in user mode and have unprocessed files, enable the button
    if data_source == 'user' and session_manager.has_unprocessed_files:
        return 'Process PDFs', False
    
    # If we're in user mode but no unprocessed files, show ready state but keep enabled
    if data_source == 'user' and session_manager.current_session_id:
        session = session_manager.sessions.get(session_manager.current_session_id, {})
        if session.get('insight_count', 0) > 0:
            return 'Ready to Reprocess', False
    
    # Default state
    if not NLP_AVAILABLE:
        return 'Process PDFs (NLP unavailable)', True
    return 'Process PDFs', not NLP_AVAILABLE

@app.callback(
    Output('upload-pdfs', 'contents'),
    Output('upload-pdfs', 'filename'),
    Output('data-refresh-trigger', 'data', allow_duplicate=True),
    Input('upload-pdfs', 'contents'),
    State('upload-pdfs', 'filename'),
    State('upload-mode', 'value'),
    State('data-source-toggle', 'value'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filenames, upload_mode, data_source):
    """Handle PDF file uploads"""
    if not contents or not filenames:
        return None, None, no_update
    
    print(f"DEBUG: Handling file upload - {len(filenames)} files")
    
    # Ensure we have lists (Dash sometimes provides single items)
    if not isinstance(contents, list):
        contents = [contents]
    if not isinstance(filenames, list):
        filenames = [filenames]
    
    if data_source != 'user':
        # If in default mode but user uploads, switch to user mode
        session_manager.current_mode = 'user'
        print("DEBUG: Switched to user mode for upload")
    
    if upload_mode == 'new' or not session_manager.current_session_id:
        # Create new session
        session_id = session_manager.create_new_session()
        print(f"DEBUG: Created new session {session_id}")
    else:
        # Use existing session
        session_id = session_manager.current_session_id
        print(f"DEBUG: Using existing session {session_id}")
    
    # Save files to session
    success, message = session_manager.add_files_to_session(session_id, contents, filenames)
    
    print(f"DEBUG: Upload result: {message}")
    
    # Clear upload area after processing and trigger refresh
    return None, None, str(time.time())

@app.callback(
    Output('process-button', 'children', allow_duplicate=True),
    Output('process-button', 'disabled', allow_duplicate=True),
    Output('data-refresh-trigger', 'data'),
    Input('process-button', 'n_clicks'),
    State('data-source-toggle', 'value'),
    prevent_initial_call=True
)
def process_uploaded_pdfs(n_clicks, data_source):
    """Process uploaded PDFs when button is clicked - REUSABLE VERSION"""
    if n_clicks == 0 or data_source != 'user':
        return 'Process PDFs', not NLP_AVAILABLE, no_update
    
    if not session_manager.current_session_id:
        return 'No session found', True, no_update
    
    print(f"DEBUG: Processing PDFs for session {session_manager.current_session_id}")
    
    # Process PDFs
    success, message = session_manager.process_session_pdfs(session_manager.current_session_id)
    
    if success:
        print("DEBUG: PDF processing completed successfully")
        # Keep button enabled but show success message
        return 'Processing Complete! Click to Reprocess', False, str(time.time())
    else:
        print(f"DEBUG: PDF processing failed: {message}")
        return f'{message}. Click to Retry', False, no_update

@app.callback(
    Output("content", "children"), 
    Input("tabs", "value"),
    Input('data-refresh-trigger', 'data')
)
def render_tab(tab, refresh_data):
    """Render the selected tab content with premium styling"""
    print(f"DEBUG: Rendering tab {tab}")
    
    # Load current data based on session mode
    ai, domain, temporal, pattern, entity = load_current_data()
    
    # Custom color scale for plots
    plot_colors = [COLORS['sidebar_accent'], COLORS['accent_light'], '#E67E22', '#9B59B6', '#34495E']
    
    # TAB 1 — Sentence Explorer (KEEP - but with improved styling)
    if tab == "ai":
        if ai.empty:
            return html.Div([
                html.H3("No Data Available", style={'color': COLORS['text_light'], 'marginBottom': '10px'}),
                html.P("Upload and process PDFs to explore AI insights", style={'color': COLORS['text_light'], 'opacity': '0.7'})
            ], style={'textAlign': 'center', 'padding': '80px 50px', 'backgroundColor': COLORS['card_bg'], 'borderRadius': '15px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.08)'})
            
        controls = html.Div([
            create_section_header("Filter Insights", "Refine your analysis with specific criteria"),
            html.Label("Domain:", style={'color': COLORS['text_dark'], 'fontWeight': '600', 'marginBottom': '8px'}), 
            dcc.Dropdown(
                options=[{"label": d, "value": d} for d in sorted(ai["domain"].dropna().unique())],
                id="domain_drop",
                style={
                    'marginBottom': '20px',
                    'backgroundColor': COLORS['card_bg'],
                    'color': COLORS['text_dark'],
                    'border': f'1px solid {COLORS["main_bg"]}',
                    'borderRadius': '8px'
                }
            ),
            html.Label("Impact Type:", style={'color': COLORS['text_dark'], 'fontWeight': '600', 'marginBottom': '8px'}), 
            dcc.Dropdown(
                options=[{"label": d, "value": d} for d in sorted(ai["impact_type"].dropna().unique())],
                id="impact_drop",
                style={
                    'marginBottom': '20px',
                    'backgroundColor': COLORS['card_bg'],
                    'color': COLORS['text_dark'],
                    'border': f'1px solid {COLORS["main_bg"]}',
                    'borderRadius': '8px'
                }
            ),
            html.Label("Timeframe:", style={'color': COLORS['text_dark'], 'fontWeight': '600', 'marginBottom': '8px'}), 
            dcc.Dropdown(
                options=[{"label": d, "value": d} for d in sorted(ai["timeframes"].dropna().unique())],
                id="time_drop",
                style={
                    'marginBottom': '20px',
                    'backgroundColor': COLORS['card_bg'],
                    'color': COLORS['text_dark'],
                    'border': f'1px solid {COLORS["main_bg"]}',
                    'borderRadius': '8px'
                }
            ),
        ], style={
            "width": "30%", 
            "float": "left", 
            "padding": "25px",
            "backgroundColor": COLORS['card_bg'],
            "borderRadius": "15px",
            "marginRight": "20px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.08)"
        })

        scatter = dcc.Graph(
            id="ai_scatter",
            style={'backgroundColor': 'transparent'}
        )
        table = dash_table.DataTable(
            id="ai_table", 
            page_size=8, 
            style_table={
                "overflowX": "auto",
                "backgroundColor": "transparent",
                "borderRadius": "10px"
            },
            style_cell={
                "textAlign": "left", 
                "padding": "12px",
                "backgroundColor": COLORS['card_bg'],
                "color": COLORS['text_dark'],
                "border": f'1px solid {COLORS["main_bg"]}'
            },
            style_header={
                'backgroundColor': COLORS['sidebar_accent'],
                'fontWeight': '700',
                'color': COLORS['sidebar_dark'],
                'border': f'1px solid {COLORS["main_bg"]}'
            },
            style_data={
                'border': f'1px solid {COLORS["main_bg"]}'
            }
        )
        
        content = html.Div([
            controls, 
            html.Div([
                create_section_header("AI Insights Explorer", "Interactive visualization of extracted insights"),
                html.Div([
                    scatter
                ], style={
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)',
                    'marginBottom': '20px'
                }),
                html.Div([
                    create_section_header("Detailed View", "Browse individual insights and their metadata"),
                    table
                ], style={
                    "marginTop": "30px",
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)'
                })
            ], style={"width": "65%", "float": "right"})
        ])
        
        return html.Div([content], style={'clear': 'both'})

    # TAB 2 — Domain Analysis (SIMPLIFIED - removed radar chart)
    if tab == "domain":
        if domain.empty:
            return html.Div("No domain data available.", style={'textAlign': 'center', 'padding': '50px', 'color': COLORS['text_light']})
        
        # 1. Stacked Bar Chart (KEEP)
        bar_fig = px.bar(domain, x="domain", y="insight_count", color="impact_type", 
                        title="Insight Distribution by Domain and Impact Type", 
                        barmode="stack", color_discrete_sequence=plot_colors)
        bar_fig.update_layout(
            plot_bgcolor=COLORS['card_bg'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text_dark'],
            title_font_color=COLORS['text_dark'],
            title_font_size=20,
            showlegend=True,
            height=500
        )
        
        # 2. Treemap for Domain Impact (KEEP)
        if 'insight_count' in domain.columns:
            fig_treemap = px.treemap(domain, path=['domain', 'impact_type'], values='insight_count',
                                   color='avg_ai_similarity', color_continuous_scale='Viridis',
                                   title="Domain Impact Treemap")
            fig_treemap.update_layout(
                plot_bgcolor=COLORS['card_bg'],
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text_dark'],
                title_font_color=COLORS['text_dark'],
                height=500
            )
        else:
            fig_treemap = go.Figure()
            fig_treemap.update_layout(
                title="No sufficient data for treemap visualization",
                plot_bgcolor=COLORS['card_bg'],
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text_dark']
            )
        
        return html.Div([
            create_section_header("Domain Analysis", "Comprehensive domain impact assessment"),
            html.Div([
                html.Div([
                    dcc.Graph(figure=bar_fig)
                ], style={
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)',
                    'marginBottom': '20px'
                }),
                html.Div([
                    dcc.Graph(figure=fig_treemap)
                ], style={
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)'
                })
            ])
        ])

    # TAB 3 — Temporal Trends (ENHANCED - using available columns)
    if tab == "temporal":
        if temporal.empty:
            return html.Div("No temporal data available.", style={'textAlign': 'center', 'padding': '50px', 'color': COLORS['text_light']})
        
        # 1. Multi-line Domain Trends with available data
        if 'dominant_domain' in temporal.columns and 'timeframe' in temporal.columns and 'avg_magnitude' in temporal.columns:
            fig_multi_line = px.line(temporal, x="timeframe", y="avg_magnitude", color="dominant_domain",
                                   title="AI Impact Magnitude Trends Across Domains", markers=True,
                                   color_discrete_sequence=plot_colors)
            fig_multi_line.update_layout(
                plot_bgcolor=COLORS['card_bg'],
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text_dark'],
                title_font_color=COLORS['text_dark'],
                height=500,
                showlegend=True,
                xaxis_title="Timeframe",
                yaxis_title="Average Impact Magnitude"
            )
        else:
            fig_multi_line = go.Figure()
            fig_multi_line.update_layout(
                title="No temporal trend data available",
                plot_bgcolor=COLORS['card_bg'],
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text_dark']
            )
        
        # 2. Area Chart for Insight Count Over Time
        if 'timeframe' in temporal.columns and 'insight_count' in temporal.columns:
            fig_area = px.area(temporal, x="timeframe", y="insight_count", 
                             title="Insight Volume Over Time",
                             color_discrete_sequence=[COLORS['sidebar_accent']])
            fig_area.update_layout(
                plot_bgcolor=COLORS['card_bg'],
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text_dark'],
                title_font_color=COLORS['text_dark'],
                height=400,
                xaxis_title="Timeframe",
                yaxis_title="Number of Insights"
            )
            
            # Add dominant impact type as line if available
            if 'dominant_impact' in temporal.columns:
                # Create a separate trace for dominant impact
                impact_counts = temporal.groupby(['timeframe', 'dominant_impact']).size().reset_index(name='count')
                fig_area.add_trace(go.Scatter(
                    x=impact_counts['timeframe'],
                    y=impact_counts['count'],
                    mode='lines+markers',
                    name='Impact Type Distribution',
                    line=dict(color=COLORS['warning'], dash='dot')
                ))
        else:
            fig_area = go.Figure()
            fig_area.update_layout(
                title="No insight count data available",
                plot_bgcolor=COLORS['card_bg'],
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text_dark']
            )
        
        return html.Div([
            create_section_header("Temporal Trends", "Evolution of AI impact across time periods"),
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_multi_line)
                ], style={
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)',
                    'marginBottom': '20px'
                }),
                html.Div([
                    dcc.Graph(figure=fig_area)
                ], style={
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)'
                })
            ])
        ])

    # TAB 4 — Pattern Insights (OK - with minor styling improvements)
    if tab == "pattern":
        if pattern.empty:
            return html.Div("No pattern data available.", style={'textAlign': 'center', 'padding': '50px', 'color': COLORS['text_light']})
        
        # Bubble chart
        bubble = px.scatter(pattern, x="pattern_type", y="avg_ai_similarity", size="insight_count", 
                           color="domains_found", hover_data=["success_rate"], 
                           title="Pattern Type Analysis", color_continuous_scale='Viridis')
        bubble.update_layout(
            plot_bgcolor=COLORS['card_bg'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text_dark'],
            title_font_color=COLORS['text_dark'],
            height=500
        )
        
        # Bar chart
        bar = px.bar(pattern, x="pattern_type", y="success_rate", title="Pattern Success Rates",
                    color_discrete_sequence=[COLORS['sidebar_accent']])
        bar.update_layout(
            plot_bgcolor=COLORS['card_bg'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text_dark'],
            title_font_color=COLORS['text_dark'],
            height=400
        )
        
        return html.Div([
            create_section_header("Pattern Insights", "Analysis of recurring patterns and their effectiveness"),
            html.Div([
                html.Div([
                    dcc.Graph(figure=bubble)
                ], style={
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)',
                    'marginBottom': '20px'
                }),
                html.Div([
                    dcc.Graph(figure=bar)
                ], style={
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)'
                })
            ])
        ])

    # TAB 5 — Entity Analysis (ENHANCED - side by side layout with magnitude)
    if tab == "entity":
        if entity.empty:
            return html.Div("No entity data available.", style={'textAlign': 'center', 'padding': '50px', 'color': COLORS['text_light']})
        
        # Word Cloud (KEEP)
        wc_img = make_wordcloud(entity, "affected_entity")
        wc_fig = html.Img(src=wc_img, style={"width": "100%", "height": "auto", "borderRadius": "10px"})
        
        # Enhanced Horizontal Bar Chart with Magnitude
        if 'mention_count' in entity.columns and 'affected_entity' in entity.columns:
            top_entities = entity.nlargest(15, 'mention_count')
            
            # Create horizontal bar chart with mention count and magnitude
            fig_bar = go.Figure()
            
            # Add mention count bars
            fig_bar.add_trace(go.Bar(
                y=top_entities['affected_entity'],
                x=top_entities['mention_count'],
                name='Mention Count',
                orientation='h',
                marker_color=COLORS['sidebar_accent'],
                text=top_entities['mention_count'],
                textposition='auto',
            ))
            
            # Add magnitude as line if available
            if 'avg_magnitude_when_mentioned' in entity.columns:
                fig_bar.add_trace(go.Scatter(
                    y=top_entities['affected_entity'],
                    x=top_entities['avg_magnitude_when_mentioned'] * (top_entities['mention_count'].max() / top_entities['avg_magnitude_when_mentioned'].max() if top_entities['avg_magnitude_when_mentioned'].max() > 0 else 1),
                    name='Average Magnitude (scaled)',
                    mode='markers+lines',
                    marker=dict(color=COLORS['warning'], size=8),
                    line=dict(color=COLORS['warning'], width=2)
                ))
            
            fig_bar.update_layout(
                title="Top Entities by Mention Count with Impact Magnitude",
                plot_bgcolor=COLORS['card_bg'],
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text_dark'],
                title_font_color=COLORS['text_dark'],
                height=600,
                showlegend=True,
                xaxis_title="Mention Count / Scaled Magnitude",
                yaxis_title="Entities",
                barmode='overlay'
            )
        else:
            fig_bar = go.Figure()
            fig_bar.update_layout(
                title="No entity mention data available",
                plot_bgcolor=COLORS['card_bg'],
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text_dark']
            )
        
        return html.Div([
            create_section_header("Entity Analysis", "Key entities and their impact relationships"),
            html.Div([
                html.Div([
                    html.H4("Entity Word Cloud", style={'color': COLORS['text_dark'], 'marginBottom': '15px', 'textAlign': 'center'}),
                    wc_fig
                ], style={
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)',
                    'width': '48%',
                    'display': 'inline-block',
                    'verticalAlign': 'top'
                }),
                html.Div([
                    dcc.Graph(figure=fig_bar)
                ], style={
                    'backgroundColor': COLORS['card_bg'],
                    'padding': '20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.08)',
                    'width': '48%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginLeft': '4%'
                })
            ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'})
        ])

# Sub-callbacks for Sentence Explorer filters
@app.callback(
    Output("ai_scatter", "figure"),
    Output("ai_table", "data"),
    Output("ai_table", "columns"),
    Input("domain_drop", "value"),
    Input("impact_drop", "value"),
    Input("time_drop", "value"),
    Input('data-refresh-trigger', 'data')
)
def update_ai(domain_v, impact_v, time_v, refresh_data):
    """Update sentence explorer based on filters"""
    ai, _, _, _, _ = load_current_data()
    
    if ai.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            plot_bgcolor=COLORS['card_bg'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text_dark']
        )
        return empty_fig, [], []
    
    df = ai.copy()
    if domain_v: 
        df = df[df["domain"] == domain_v]
    if impact_v: 
        df = df[df["impact_type"] == impact_v]
    if time_v: 
        df = df[df["timeframes"] == time_v]
    
    fig = px.scatter(df, x="ai_similarity", y="sentence_length", color="impact_type", 
                    hover_data=["domain", "sentiment_score", "sentence"], 
                    title="AI Similarity vs Sentence Length",
                    color_discrete_sequence=[COLORS['sidebar_accent'], COLORS['accent_light'], COLORS['warning']])
    
    fig.update_layout(
        plot_bgcolor=COLORS['card_bg'],
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text_dark'],
        title_font_color=COLORS['text_dark']
    )
    
    cols = [{"name": c, "id": c} for c in ["sentence", "domain", "impact_type", "sentiment_score", "timeframes"]]
    data = df[["sentence", "domain", "impact_type", "sentiment_score", "timeframes"]].to_dict("records")
    
    return fig, data, cols

# ---------- Run ----------
if __name__ == '__main__':
    print("Starting Chrono-Insight Dashboard...")
    print("Features: Professional Design, Enhanced Analytics, Clean Interface")
    print("Enhanced Visualizations: Multi-line trends, Entity magnitude analysis, Domain treemaps")
    print("DEBUG: Debug mode enabled - check console for detailed logs")
    app.run(debug=True, use_reloader=False, port=8010)