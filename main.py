import streamlit as st
import json
import time
from datetime import datetime, timedelta
from groq import Groq
import re
from typing import List, Dict, Any
import uuid

# Initialize Groq client


def init_groq_client(api_key: str):
    return Groq(api_key=api_key)


class AIInterviewer:
    def __init__(self, groq_client):
        self.client = groq_client
        self.conversation_history = []
        self.interview_phases = [
            "introduction",
            "technical_deep_dive",
            "experience_exploration",
            "problem_solving",
            "behavioral_questions",
            "scenario_based",
            "closing"
        ]
        self.current_phase = 0
        self.start_time = None
        self.target_duration = 40 * 60  # 40 minutes in seconds

    def analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """Analyze resume to extract key information for interview preparation"""

        prompt = f"""
        Analyze this resume and extract key information for conducting a technical interview:
        
        Resume:
        {resume_text}
        
        Please provide a JSON response with:
        1. candidate_name: extracted name
        2. experience_level: junior/mid/senior based on years of experience
        3. key_skills: list of technical skills mentioned
        4. programming_languages: list of programming languages
        5. projects: list of notable projects with brief descriptions
        6. work_experience: list of companies and roles
        7. education: educational background
        8. certifications: any certifications mentioned
        9. suggested_focus_areas: areas to focus the interview on
        10. estimated_experience_years: number of years of experience
        
        Respond only with valid JSON.
        """

        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )

            content = response.choices[0].message.content
            # Extract JSON from response if it's wrapped in text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(content)

        except Exception as e:
            st.error(f"Error analyzing resume: {e}")
            return self._get_default_analysis()

    def _get_default_analysis(self):
        return {
            "candidate_name": "Candidate",
            "experience_level": "mid",
            "key_skills": ["Python", "JavaScript"],
            "programming_languages": ["Python"],
            "projects": ["Web application"],
            "work_experience": ["Software Developer"],
            "education": ["Computer Science"],
            "certifications": [],
            "suggested_focus_areas": ["Programming", "Problem Solving"],
            "estimated_experience_years": 3
        }

    def generate_question(self, resume_analysis: Dict, conversation_context: List) -> str:
        """Generate contextually appropriate interview question"""

        phase_prompts = {
            "introduction": f"""
            You are an AI interviewer conducting a professional technical interview. 
            Generate a warm, professional introduction question for {resume_analysis.get('candidate_name', 'the candidate')}.
            
            Consider their background: {resume_analysis.get('experience_level')} level with skills in {', '.join(resume_analysis.get('key_skills', [])[:3])}.
            
            Start with a brief introduction of yourself as an AI interviewer, then ask them to introduce themselves and walk through their background.
            Keep it conversational and professional.
            """,

            "technical_deep_dive": f"""
            Based on the candidate's background in {', '.join(resume_analysis.get('key_skills', [])[:5])}, 
            generate a technical question that explores their depth of knowledge.
            
            Focus on: {', '.join(resume_analysis.get('suggested_focus_areas', [])[:3])}
            Experience level: {resume_analysis.get('experience_level')}
            
            Make it challenging but appropriate for their level. Ask about concepts, best practices, or implementation details.
            """,

            "experience_exploration": f"""
            Ask about their specific work experience and projects. 
            They have worked on: {', '.join([p.get('name', p.get('description', 'a project')) for p in resume_analysis.get('projects', [])[:2]])}
            Work experience: {', '.join(resume_analysis.get('work_experience', [])[:2])}
            
            Ask them to elaborate on a specific project or role, focusing on challenges they faced and how they solved them.
            """,

            "problem_solving": """
            Present a practical coding or system design problem that requires analytical thinking.
            Make it relevant to their skill set but challenging enough to assess problem-solving approach.
            Ask them to walk through their thought process step by step.
            """,

            "behavioral_questions": """
            Ask a behavioral question about teamwork, leadership, conflict resolution, or handling difficult situations.
            Use the STAR method framework (Situation, Task, Action, Result) and ask for specific examples.
            """,

            "scenario_based": f"""
            Present a real-world scenario related to {resume_analysis.get('experience_level')} level work in {', '.join(resume_analysis.get('key_skills', [])[:2])}.
            Ask how they would approach it, what considerations they would have, and what potential challenges they foresee.
            """,

            "closing": """
            Generate a closing question for the interview. Ask if they have any questions about the role, 
            company, or interview process. Also ask about their career goals and what they're looking for in their next opportunity.
            """
        }

        current_phase_name = self.interview_phases[self.current_phase]
        base_prompt = phase_prompts.get(
            current_phase_name, phase_prompts["technical_deep_dive"])

        # Add conversation context
        context = ""
        if conversation_context:
            recent_context = conversation_context[-3:]  # Last 3 exchanges
            context = f"""
            Recent conversation context:
            {json.dumps(recent_context, indent=2)}
            
            Build upon this conversation naturally. Don't repeat similar questions.
            """

        full_prompt = f"""
        {base_prompt}
        
        {context}
        
        Current interview phase: {current_phase_name}
        Time elapsed: {self._get_elapsed_time()} minutes
        
        Generate ONE specific, engaging question. Be conversational but professional.
        The question should take 2-4 minutes to answer thoroughly.
        """

        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            st.error(f"Error generating question: {e}")
            return self._get_fallback_question(current_phase_name)

    def _get_fallback_question(self, phase: str):
        fallback_questions = {
            "introduction": "Could you please introduce yourself and tell me about your background?",
            "technical_deep_dive": "Can you explain a complex technical concept you've worked with recently?",
            "experience_exploration": "Tell me about a challenging project you've worked on and how you approached it.",
            "problem_solving": "How would you approach debugging a performance issue in a web application?",
            "behavioral_questions": "Describe a time when you had to work with a difficult team member. How did you handle it?",
            "scenario_based": "If you were tasked with improving the performance of a slow database query, what steps would you take?",
            "closing": "Do you have any questions for me about the role or the company?"
        }
        return fallback_questions.get(phase, "Can you tell me more about your experience?")

    def should_continue_interview(self) -> bool:
        """Determine if interview should continue based on time and phases"""
        if not self.start_time:
            return True

        elapsed = time.time() - self.start_time

        # Continue if we haven't reached target duration and haven't finished all phases
        return elapsed < self.target_duration and self.current_phase < len(self.interview_phases) - 1

    def advance_phase(self):
        """Move to next interview phase"""
        if self.current_phase < len(self.interview_phases) - 1:
            self.current_phase += 1

    def _get_elapsed_time(self) -> int:
        """Get elapsed time in minutes"""
        if not self.start_time:
            return 0
        return int((time.time() - self.start_time) / 60)

    def start_interview(self):
        """Initialize interview session"""
        self.start_time = time.time()
        self.conversation_history = []
        self.current_phase = 0

# New function for response actions


def display_response_actions(interviewer: AIInterviewer, current_exchange: Dict[str, Any], response_key: str):
    response = st.text_area(
        "Response",
        height=200,
        key=response_key,
        placeholder="""Provide a detailed response here...

Tips for a strong answer:
• Explain your thought process step by step
• Use specific examples from your experience
• Mention relevant technologies or frameworks
• Discuss challenges and how you overcame them
• Be honest about what you know and don't know""",
        label_visibility="collapsed"
    )

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("↩️ Go Back", use_container_width=True):
            if len(st.session_state.conversation_history) > 1:
                # Remove current incomplete exchange
                st.session_state.conversation_history.pop()
                # Optional: Potentially revert to previous phase if desired,
                # but usually interview phases advance regardless of going back on a question.
                # interviewer.current_phase = max(0, interviewer.current_phase - 1)
                st.rerun()
            else:
                st.warning("Cannot go back further.")

    with col2:
        if st.button("➡️ Submit Response & Next Question", type="primary", use_container_width=True):
            if response.strip():
                current_exchange['response'] = response
                interviewer.advance_phase()
                st.rerun()
            else:
                st.error("Please provide a response before submitting.")

    with col3:
        if st.button("⏭️ Skip Question", use_container_width=True):
            current_exchange['response'] = "[Skipped]"
            interviewer.advance_phase()
            st.rerun()


def main():
    st.set_page_config(
        page_title="AI Technical Interviewer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .phase-indicator {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .interview-stats {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .question-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .response-area {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #667eea;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #5a6fd8;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
    }
    .ai-message {
        background-color: #f8f9ff;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0fff4;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Technical Interviewer</h1>
        <p style="font-size: 1.2em; margin: 0;">Intelligent AI-powered technical interviews tailored to your experience</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Sidebar for configuration
    with st.sidebar:
        st.markdown("### 🔧 Configuration")

        # Groq API Key input with better styling
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Enter your Groq API key to enable AI interviewer functionality",
            placeholder="Enter your API key here..."
        )

        if groq_api_key:
            st.success("✅ API Key configured successfully!")
        else:
            st.warning("⚠️ Please enter your Groq API key")
            st.info(
                "💡 Get your free API key from [Groq Console](https://console.groq.com/)")

        st.markdown("---")

        # Interview Progress Section
        st.markdown("### 📊 Interview Progress")

        if 'interviewer' in st.session_state and st.session_state.interviewer.start_time:
            elapsed = st.session_state.interviewer._get_elapsed_time()

            # Time metrics with better layout
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⏱️ Elapsed", f"{elapsed} min")
            with col2:
                remaining = max(0, 40 - elapsed)
                st.metric("⏳ Remaining", f"{remaining} min")

            # Progress bar with percentage
            progress = min(elapsed / 40, 1.0)
            st.progress(progress)
            st.caption(f"Progress: {int(progress * 100)}%")

            # Current phase indicator
            current_phase = st.session_state.interviewer.interview_phases[
                st.session_state.interviewer.current_phase]
            phase_display = current_phase.replace('_', ' ').title()

            st.markdown(f"""
            <div class="phase-indicator">
                <strong>📍 Current Phase:</strong><br>
                {phase_display}
            </div>
            """, unsafe_allow_html=True)

            # Phase progress
            phase_progress = (st.session_state.interviewer.current_phase + 1) / \
                len(st.session_state.interviewer.interview_phases)
            st.progress(phase_progress)
            st.caption(
                f"Phase {st.session_state.interviewer.current_phase + 1} of {len(st.session_state.interviewer.interview_phases)}")

        else:
            st.info("🎯 Interview not started yet")

        st.markdown("---")

        # Interview Statistics
        if 'conversation_history' in st.session_state and st.session_state.conversation_history:
            st.markdown("### 📈 Session Stats")

            total_questions = len(st.session_state.conversation_history)
            answered_questions = len([q for q in st.session_state.conversation_history if q.get(
                'response') and q['response'] != '[Skipped]'])
            skipped_questions = len(
                [q for q in st.session_state.conversation_history if q.get('response') == '[Skipped]'])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("❓ Questions", total_questions)
                st.metric("✅ Answered", answered_questions)
            with col2:
                st.metric("⏭️ Skipped", skipped_questions)
                if total_questions > 0:
                    completion_rate = (answered_questions /
                                       total_questions) * 100
                    st.metric("📊 Completion", f"{completion_rate:.0f}%")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'resume_analyzed' not in st.session_state:
        st.session_state.resume_analyzed = False
    if 'resume_analysis' not in st.session_state:
        st.session_state.resume_analysis = {}
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False

    # Main interface with enhanced error handling
    if not groq_api_key:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 2rem; border-radius: 10px; border-left: 4px solid #ffc107; margin: 2rem 0;">
            <h3>🔑 API Key Required</h3>
            <p>Please enter your Groq API key in the sidebar to begin the interview process.</p>
            <p><strong>How to get started:</strong></p>
            <ol>
                <li>Visit <a href="https://console.groq.com/" target="_blank">Groq Console</a></li>
                <li>Create a free account</li>
                <li>Generate an API key</li>
                <li>Enter it in the sidebar</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    # Initialize interviewer
    if 'interviewer' not in st.session_state:
        try:
            client = init_groq_client(groq_api_key)
            st.session_state.interviewer = AIInterviewer(client)
        except Exception as e:
            st.error(f"Error initializing AI interviewer: {e}")
            return

    # Resume upload and analysis with enhanced UI
    if not st.session_state.resume_analyzed:
        st.markdown("## 📄 Resume Analysis")

        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📝 Paste Resume Text", "📁 Upload File"])

        resume_text = ""

        with tab1:
            st.markdown("### Copy and paste your resume content:")
            resume_text = st.text_area(
                "Resume Content",
                height=400,
                placeholder="""Paste your complete resume here...

Include:
• Work experience and roles
• Technical skills and technologies
• Education and certifications
• Notable projects and achievements
• Programming languages and frameworks

The more detailed your resume, the better the AI can tailor the interview!""",
                label_visibility="collapsed"
            )

            if resume_text:
                word_count = len(resume_text.split())
                st.caption(f"📊 Word count: {word_count} words")

                if word_count < 50:
                    st.warning(
                        "⚠️ Your resume seems quite short. Consider adding more details for better interview customization.")
                elif word_count > 1000:
                    st.info(
                        "📋 Comprehensive resume detected - perfect for detailed analysis!")
                else:
                    st.success("✅ Good resume length for analysis")

        with tab2:
            st.markdown("### Upload your resume file:")

            uploaded_file = st.file_uploader(
                "Choose your resume file",
                type=['txt'],
                help="Currently supports text files (.txt). PDF and DOCX support coming soon!",
                accept_multiple_files=False
            )

            if uploaded_file:
                try:
                    resume_text = str(uploaded_file.read(), "utf-8")
                    st.success(
                        f"✅ File '{uploaded_file.name}' uploaded successfully!")

                    # Show preview
                    with st.expander("📖 Preview uploaded content"):
                        st.text_area("File Content Preview", resume_text[:500] + "..." if len(
                            resume_text) > 500 else resume_text, height=150, disabled=True)

                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")

        # Analysis button with enhanced styling
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyze Resume & Start Interview", type="primary", use_container_width=True):
                if resume_text.strip():
                    with st.spinner("🔍 Analyzing your resume and preparing personalized interview questions..."):
                        # Show progress steps
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        status_text.text("📄 Processing resume content...")
                        progress_bar.progress(25)
                        time.sleep(1)

                        status_text.text(
                            "🧠 Extracting skills and experience...")
                        progress_bar.progress(50)

                        st.session_state.resume_analysis = st.session_state.interviewer.analyze_resume(
                            resume_text)

                        status_text.text("🎯 Generating interview strategy...")
                        progress_bar.progress(75)
                        time.sleep(1)

                        st.session_state.resume_analyzed = True
                        st.session_state.interviewer.start_interview()
                        st.session_state.interview_started = True

                        status_text.text("✅ Ready to begin interview!")
                        progress_bar.progress(100)
                        time.sleep(1)

                        st.rerun()
                else:
                    st.error(
                        "❌ Please provide your resume content before starting the interview.")

        # Instructions section
        with st.expander("💡 Tips for Best Results", expanded=False):
            st.markdown("""
            **For the most effective interview experience:**
            
            ✅ **Include comprehensive details:**
            - All relevant work experience with specific technologies used
            - Complete list of programming languages and frameworks
            - Notable projects with brief descriptions
            - Education and certifications, and achievements
            
            ✅ **Be specific about your experience:**
            - Mention years of experience with different technologies
            - Include both technical and soft skills
            - Add context about team sizes and project scope
            
            ✅ **Keep it current:**
            - Focus on recent and relevant experience
            - Include your most impressive achievements
            - Mention any leadership or mentoring experience
            
            The AI will use this information to create a personalized 40+ minute interview experience!
            """)

    # Enhanced Interview interface
    if st.session_state.resume_analyzed and st.session_state.interview_started:
        st.markdown("## 🎤 Interview Session")

        # Display enhanced resume analysis summary
        with st.expander("📊 Resume Analysis Summary", expanded=False):
            analysis = st.session_state.resume_analysis

            # Overview metrics
            st.markdown("### 👤 Candidate Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                experience_level = analysis.get(
                    'experience_level', 'N/A').title()
                level_emoji = {"Junior": "🌱", "Mid": "🌿",
                               "Senior": "🌳"}.get(experience_level, "📊")
                st.metric(f"{level_emoji} Experience Level", experience_level)

            with col2:
                years = analysis.get('estimated_experience_years', 'N/A')
                st.metric("📅 Years Experience",
                          f"{years}" if years != 'N/A' else 'N/A')

            with col3:
                skills_count = len(analysis.get('key_skills', []))
                st.metric("🛠️ Key Skills", skills_count)

            with col4:
                projects_count = len(analysis.get('projects', []))
                st.metric("📁 Projects", projects_count)

            # Detailed breakdown
            col1, col2 = st.columns(2)

            with col1:
                if analysis.get('key_skills'):
                    st.markdown("**🔧 Technical Skills:**")
                    skills_display = ", ".join(
                        analysis.get('key_skills', [])[:8])
                    if len(analysis.get('key_skills', [])) > 8:
                        skills_display += f" + {len(analysis.get('key_skills', [])) - 8} more"
                    st.markdown(f"*{skills_display}*")

                if analysis.get('programming_languages'):
                    st.markdown("**💻 Programming Languages:**")
                    st.markdown(
                        f"*{', '.join(analysis.get('programming_languages', []))}*")

            with col2:
                if analysis.get('suggested_focus_areas'):
                    st.markdown("**🎯 Interview Focus Areas:**")
                    for area in analysis.get('suggested_focus_areas', [])[:5]:
                        st.markdown(f"• {area}")

                if analysis.get('work_experience'):
                    st.markdown("**💼 Recent Experience:**")
                    for exp in analysis.get('work_experience', [])[:3]:
                        st.markdown(f"• {exp}")

        # Interview statistics dashboard
        if st.session_state.conversation_history:
            st.markdown("""
            <div class="interview-stats">
                <h4>📈 Interview Progress Dashboard</h4>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            total_q = len(st.session_state.conversation_history)
            answered_q = len([q for q in st.session_state.conversation_history if q.get(
                'response') and q['response'] != '[Skipped]'])
            skipped_q = len([q for q in st.session_state.conversation_history if q.get(
                'response') == '[Skipped]'])

            with col1:
                st.metric("📝 Total Questions", total_q)
            with col2:
                st.metric("✅ Answered", answered_q)
            with col3:
                st.metric("⏭️ Skipped", skipped_q)
            with col4:
                if total_q > 0:
                    completion = (answered_q / total_q) * 100
                    st.metric("🎯 Response Rate", f"{completion:.0f}%")

        st.markdown("---")

        # Chat interface with enhanced styling
        st.markdown("### 💬 Interview Conversation")

        # Create a scrollable chat container
        chat_container = st.container()

        with chat_container:
            for i, exchange in enumerate(st.session_state.conversation_history):
                # AI Question with enhanced styling
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5em; margin-right: 0.5rem;">🤖</span>
                        <strong>AI Interviewer</strong>
                        <span style="margin-left: auto; color: #6c757d; font-size: 0.9em;">
                            Question {i+1} • {exchange.get('phase', 'interview').replace('_', ' ').title()}
                        </span>
                    </div>
                    <div style="margin-left: 2rem;">
                        {exchange['question']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # User Response with enhanced styling
                if exchange.get('response'):
                    response_text = exchange['response']
                    response_color = "#d4edda" if response_text != "[Skipped]" else "#f8d7da"
                    response_icon = "👤" if response_text != "[Skipped]" else "⏭️"
                    response_label = "Your Response" if response_text != "[Skipped]" else "Skipped Question"

                    st.markdown(f"""
                    <div class="chat-message user-message" style="background-color: {response_color};">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.5em; margin-right: 0.5rem;">{response_icon}</span>
                            <strong>{response_label}</strong>
                        </div>
                        <div style="margin-left: 2rem;">
                            {response_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Add spacing between exchanges
                st.markdown("<br>", unsafe_allow_html=True)

        # Current question interface with enhanced UX
        if st.session_state.interviewer.should_continue_interview():
            # Generate new question if needed
            if not st.session_state.conversation_history or st.session_state.conversation_history[-1].get('response'):
                with st.spinner("🤔 AI is preparing your next question..."):
                    question = st.session_state.interviewer.generate_question(
                        st.session_state.resume_analysis,
                        st.session_state.conversation_history
                    )

                    st.session_state.conversation_history.append({
                        'question': question,
                        'response': None,
                        'timestamp': datetime.now().isoformat(),
                        'phase': st.session_state.interviewer.interview_phases[st.session_state.interviewer.current_phase]
                    })
                    st.rerun()

            # Show current question with enhanced styling
            current_exchange = st.session_state.conversation_history[-1]

            if not current_exchange.get('response'):
                # Current question display
                current_phase = current_exchange.get(
                    'phase', 'interview').replace('_', ' ').title()
                question_number = len(st.session_state.conversation_history)

                st.markdown(f"""
                <div class="question-box">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 2em; margin-right: 1rem;">🤖</span>
                        <div>
                            <h4 style="margin: 0; color: #667eea;">AI Interviewer</h4>
                            <small style="color: #6c757d;">Question {question_number} • {current_phase} Phase</small>
                        </div>
                    </div>
                    <div style="font-size: 1.1em; line-height: 1.6; margin-left: 3rem;">
                        {current_exchange['question']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Response input area with enhanced styling
                st.markdown("""
                <div class="response-area">
                    <h5>💭 Your Response:</h5>
                    <p style="color: #6c757d; margin-bottom: 1rem;">
                        Take your time to provide a thoughtful, detailed response. The AI is looking for your reasoning process and technical understanding.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Call the new function for response actions
                display_response_actions(
                    st.session_state.interviewer,
                    current_exchange,
                    f"response_{len(st.session_state.conversation_history)}"
                )
        else:
            st.success("🎉 Interview Completed! Thank you for participating.")
            st.info(
                "You can refresh the page to start a new interview or review the conversation history above.")
            st.markdown("---")
            if st.button("Start New Interview", type="primary"):
                st.session_state.clear()
                st.rerun()


if __name__ == "__main__":
    main()
