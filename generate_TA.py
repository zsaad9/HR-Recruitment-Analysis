#!/usr/bin/env python3
"""
Talent Acquisition Analytics - Dataset Generator
Creates realistic TA datasets for analytics project
Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

print("🚀 Starting Talent Acquisition Dataset Generation...")
print("=" * 60)

# Configuration
NUM_APPLICATIONS = 2847
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 6, 30)

# Helper functions
def random_date(start, end):
    """Generate random date between start and end"""
    delta = end - start
    int_delta = delta.days
    random_day = random.randrange(int_delta)
    return start + timedelta(days=random_day)

def weighted_choice(choices, weights):
    """Make weighted random choice with automatic normalization"""
    # Normalize weights to ensure they sum to 1.0
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return np.random.choice(choices, p=weights)

# Define realistic data parameters
job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'Sales Representative', 
              'Marketing Manager', 'UX Designer', 'DevOps Engineer', 'Business Analyst',
              'Customer Success Manager', 'HR Generalist']

departments = ['Engineering', 'Data Science', 'Product', 'Sales', 'Marketing', 
               'Design', 'Operations', 'Analytics', 'Customer Success', 'Human Resources']

locations = ['New York, NY', 'San Francisco, CA', 'Austin, TX', 'Seattle, WA', 'Chicago, IL',
            'Boston, MA', 'Los Angeles, CA', 'Denver, CO', 'Atlanta, GA', 'Remote']

sources = ['LinkedIn', 'Indeed', 'Company Website', 'Employee Referral', 'Recruiter Outreach',
          'University Career Fair', 'Glassdoor', 'AngelList', 'Stack Overflow', 'Networking Event',
          'Social Media', 'Other']

source_weights = [0.15, 0.12, 0.18, 0.08, 0.10, 0.05, 0.08, 0.04, 0.03, 0.05, 0.07, 0.05]

experience_levels = ['Entry Level', 'Mid Level', 'Senior Level', 'Executive']
experience_weights = [0.25, 0.45, 0.25, 0.05]

education_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
education_weights = [0.05, 0.10, 0.50, 0.30, 0.05]

ethnicities = ['White', 'Asian', 'Black or African American', 'Hispanic or Latino', 
               'Native American', 'Two or More Races', 'Prefer not to say']
ethnicity_weights = [0.54, 0.20, 0.12, 0.08, 0.02, 0.04, 0.00]  # Sums to 1.0

genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
gender_weights = [0.52, 0.40, 0.03, 0.05]

# Stage progression probabilities (funnel conversion rates)
stage_progression = {
    'Application': 1.0,
    'Phone Screen': 0.35,
    'Technical Assessment': 0.65,
    'First Interview': 0.70,
    'Final Interview': 0.75,
    'Reference Check': 0.85,
    'Offer Extended': 0.90,
    'Hired': 0.78
}

stages = list(stage_progression.keys())

print("📝 Generating Applications Dataset...")

# Generate Applications Dataset
applications_data = []
candidate_id = 1

for i in range(NUM_APPLICATIONS):
    if i % 500 == 0:
        print(f"   Progress: {i}/{NUM_APPLICATIONS} applications...")
        
    application_date = random_date(START_DATE, END_DATE)
    job_title = random.choice(job_titles)
    department = departments[job_titles.index(job_title)]
    location = random.choice(locations)
    source = weighted_choice(sources, source_weights)
    
    # Demographics with some correlation to source
    if source == 'University Career Fair':
        experience_level = weighted_choice(experience_levels, [0.70, 0.25, 0.05, 0.0])
        education_level = weighted_choice(education_levels, [0.0, 0.05, 0.70, 0.25, 0.0])
    elif source == 'Employee Referral':
        experience_level = weighted_choice(experience_levels, [0.15, 0.50, 0.30, 0.05])
        education_level = weighted_choice(education_levels, [0.02, 0.08, 0.45, 0.40, 0.05])
    else:
        experience_level = weighted_choice(experience_levels, experience_weights)
        education_level = weighted_choice(education_levels, education_weights)
    
    # Generate candidate info
    gender = weighted_choice(genders, gender_weights)
    first_name = fake.first_name_male() if gender == 'Male' else fake.first_name_female()
    if gender in ['Non-binary', 'Prefer not to say']:
        first_name = fake.first_name()
    
    last_name = fake.last_name()
    email = f"{first_name.lower()}.{last_name.lower()}@{fake.domain_name()}"
    
    ethnicity = weighted_choice(ethnicities, ethnicity_weights)
    
    # Salary expectations based on role and experience
    salary_base = {
        'Software Engineer': {'Entry Level': 75000, 'Mid Level': 105000, 'Senior Level': 140000, 'Executive': 200000},
        'Data Scientist': {'Entry Level': 80000, 'Mid Level': 115000, 'Senior Level': 150000, 'Executive': 220000},
        'Product Manager': {'Entry Level': 70000, 'Mid Level': 110000, 'Senior Level': 145000, 'Executive': 210000},
        'Sales Representative': {'Entry Level': 50000, 'Mid Level': 75000, 'Senior Level': 120000, 'Executive': 180000},
        'Marketing Manager': {'Entry Level': 55000, 'Mid Level': 80000, 'Senior Level': 120000, 'Executive': 180000},
        'UX Designer': {'Entry Level': 60000, 'Mid Level': 85000, 'Senior Level': 125000, 'Executive': 175000},
        'DevOps Engineer': {'Entry Level': 70000, 'Mid Level': 100000, 'Senior Level': 135000, 'Executive': 190000},
        'Business Analyst': {'Entry Level': 55000, 'Mid Level': 80000, 'Senior Level': 115000, 'Executive': 160000},
        'Customer Success Manager': {'Entry Level': 50000, 'Mid Level': 75000, 'Senior Level': 110000, 'Executive': 155000},
        'HR Generalist': {'Entry Level': 45000, 'Mid Level': 65000, 'Senior Level': 90000, 'Executive': 140000}
    }
    
    base_salary = salary_base[job_title][experience_level]
    salary_expectation = int(base_salary * np.random.uniform(0.85, 1.15))
    
    # Years of experience
    exp_ranges = {'Entry Level': (0, 2), 'Mid Level': (3, 7), 'Senior Level': (8, 15), 'Executive': (15, 25)}
    years_experience = random.randint(*exp_ranges[experience_level])
    
    applications_data.append({
        'candidate_id': candidate_id,
        'application_date': application_date,
        'job_title': job_title,
        'department': department,
        'location': location,
        'source': source,
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'gender': gender,
        'ethnicity': ethnicity,
        'experience_level': experience_level,
        'years_experience': years_experience,
        'education_level': education_level,
        'salary_expectation': salary_expectation
    })
    
    candidate_id += 1

applications_df = pd.DataFrame(applications_data)

print("🎯 Generating Interview Pipeline Dataset...")

# Generate Interview Pipeline Dataset
pipeline_data = []
interview_id = 1

for idx, (_, app) in enumerate(applications_df.iterrows()):
    if idx % 500 == 0:
        print(f"   Progress: {idx}/{len(applications_df)} candidates...")
        
    candidate_id = app['candidate_id']
    current_date = app['application_date']
    
    # Determine how far through pipeline this candidate gets
    for stage in stages:
        # Base probability
        base_prob = stage_progression[stage]
        
        # Adjust probability based on source quality
        source_multipliers = {
            'Employee Referral': 1.3,
            'Recruiter Outreach': 1.2,
            'Company Website': 1.1,
            'LinkedIn': 1.0,
            'University Career Fair': 0.9,
            'Indeed': 0.85,
            'Glassdoor': 0.85,
            'Other': 0.7
        }
        
        prob = base_prob * source_multipliers.get(app['source'], 0.8)
        prob = min(prob, 0.95)  # Cap at 95%
        
        if random.random() <= prob:
            # Calculate stage duration based on stage type
            stage_durations = {
                'Application': (0, 1),
                'Phone Screen': (3, 10),
                'Technical Assessment': (1, 5),
                'First Interview': (2, 8),
                'Final Interview': (3, 12),
                'Reference Check': (1, 4),
                'Offer Extended': (1, 7),
                'Hired': (3, 14)
            }
            
            if stage != 'Application':
                duration_days = random.randint(*stage_durations[stage])
                current_date += timedelta(days=duration_days)
            
            # Generate stage outcome
            if stage == stages[-1] or random.random() > stage_progression.get(stages[stages.index(stage) + 1] if stage != stages[-1] else 1, 0):
                # This is the final stage for this candidate
                status = 'Completed' if stage == 'Hired' else 'Rejected'
                
                # Rejection reasons
                rejection_reasons = {
                    'Phone Screen': ['Not a culture fit', 'Insufficient experience', 'Salary expectations too high', 'Communication issues'],
                    'Technical Assessment': ['Failed technical evaluation', 'Incomplete submission', 'Did not attend', 'Below threshold score'],
                    'First Interview': ['Not a culture fit', 'Insufficient technical skills', 'Poor communication', 'Better candidate found'],
                    'Final Interview': ['Salary negotiation failed', 'Not the right fit', 'Candidate withdrew', 'Reference check issues'],
                    'Reference Check': ['Poor references', 'Verification failed', 'Background check issues'],
                    'Offer Extended': ['Candidate declined', 'Accepted counter-offer', 'Salary too low', 'Start date conflict'],
                    'Hired': ['Successful hire']
                }
                
                outcome = 'Hired' if status == 'Completed' else random.choice(rejection_reasons.get(stage, ['Other']))
            else:
                status = 'In Progress'
                outcome = 'Advancing to next stage'
            
            # Interview feedback score (1-5 scale)
            feedback_score = None
            if stage in ['Phone Screen', 'First Interview', 'Final Interview']:
                # Score influenced by whether they advance
                if status == 'In Progress' or stage == 'Hired':
                    feedback_score = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
                else:
                    feedback_score = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            
            pipeline_data.append({
                'interview_id': interview_id,
                'candidate_id': candidate_id,
                'stage': stage,
                'stage_date': current_date,
                'status': status,
                'outcome': outcome,
                'feedback_score': feedback_score,
                'interviewer': fake.name() if stage in ['Phone Screen', 'First Interview', 'Final Interview'] else None
            })
            
            interview_id += 1
            
            if status != 'In Progress':
                break
        else:
            # Rejected at this stage without proceeding
            if stage != 'Application':
                pipeline_data.append({
                    'interview_id': interview_id,
                    'candidate_id': candidate_id,
                    'stage': stage,
                    'stage_date': current_date,
                    'status': 'Rejected',
                    'outcome': 'Did not meet requirements',
                    'feedback_score': None,
                    'interviewer': None
                })
                interview_id += 1
            break

pipeline_df = pd.DataFrame(pipeline_data)

print("💰 Generating Offers Dataset...")

# Generate Offers Dataset
offers_data = []
offer_id = 1

hired_candidates = pipeline_df[pipeline_df['outcome'] == 'Hired']['candidate_id'].unique()

for candidate_id in hired_candidates:
    app_info = applications_df[applications_df['candidate_id'] == candidate_id].iloc[0]
    hire_date = pipeline_df[(pipeline_df['candidate_id'] == candidate_id) & 
                          (pipeline_df['stage'] == 'Hired')]['stage_date'].iloc[0]
    
    # Generate offer details
    salary_base = {
        'Software Engineer': {'Entry Level': 75000, 'Mid Level': 105000, 'Senior Level': 140000, 'Executive': 200000},
        'Data Scientist': {'Entry Level': 80000, 'Mid Level': 115000, 'Senior Level': 150000, 'Executive': 220000},
        'Product Manager': {'Entry Level': 70000, 'Mid Level': 110000, 'Senior Level': 145000, 'Executive': 210000},
        'Sales Representative': {'Entry Level': 50000, 'Mid Level': 75000, 'Senior Level': 120000, 'Executive': 180000},
        'Marketing Manager': {'Entry Level': 55000, 'Mid Level': 80000, 'Senior Level': 120000, 'Executive': 180000},
        'UX Designer': {'Entry Level': 60000, 'Mid Level': 85000, 'Senior Level': 125000, 'Executive': 175000},
        'DevOps Engineer': {'Entry Level': 70000, 'Mid Level': 100000, 'Senior Level': 135000, 'Executive': 190000},
        'Business Analyst': {'Entry Level': 55000, 'Mid Level': 80000, 'Senior Level': 115000, 'Executive': 160000},
        'Customer Success Manager': {'Entry Level': 50000, 'Mid Level': 75000, 'Senior Level': 110000, 'Executive': 155000},
        'HR Generalist': {'Entry Level': 45000, 'Mid Level': 65000, 'Senior Level': 90000, 'Executive': 140000}
    }
    
    offered_salary = int(salary_base[app_info['job_title']][app_info['experience_level']] * 
                        np.random.uniform(0.95, 1.10))
    
    # Benefits package value (15-25% of salary)
    benefits_value = int(offered_salary * np.random.uniform(0.15, 0.25))
    
    # Start date (2-4 weeks after offer)
    start_date = hire_date + timedelta(days=random.randint(14, 28))
    
    offers_data.append({
        'offer_id': offer_id,
        'candidate_id': candidate_id,
        'offer_date': hire_date - timedelta(days=random.randint(1, 5)),
        'offered_salary': offered_salary,
        'benefits_value': benefits_value,
        'start_date': start_date,
        'offer_accepted': True,
        'negotiation_rounds': random.randint(0, 3),
        'recruiter': fake.name()
    })
    
    offer_id += 1

# Add some declined offers
declined_candidates = pipeline_df[pipeline_df['outcome'].str.contains('declined|Accepted counter-offer', na=False, case=False)]['candidate_id'].unique()

for candidate_id in declined_candidates[:len(declined_candidates)//2]:  # Only half actually got offers
    app_info = applications_df[applications_df['candidate_id'] == candidate_id].iloc[0]
    offer_date = pipeline_df[(pipeline_df['candidate_id'] == candidate_id) & 
                           (pipeline_df['stage'] == 'Offer Extended')]['stage_date'].iloc[0]
    
    offered_salary = int(salary_base[app_info['job_title']][app_info['experience_level']] * 
                        np.random.uniform(0.90, 1.05))  # Slightly lower for declined offers
    benefits_value = int(offered_salary * np.random.uniform(0.15, 0.25))
    
    offers_data.append({
        'offer_id': offer_id,
        'candidate_id': candidate_id,
        'offer_date': offer_date,
        'offered_salary': offered_salary,
        'benefits_value': benefits_value,
        'start_date': None,
        'offer_accepted': False,
        'negotiation_rounds': random.randint(1, 4),
        'recruiter': fake.name()
    })
    
    offer_id += 1

offers_df = pd.DataFrame(offers_data)

print("👥 Generating Recruiter Performance Dataset...")

# Generate Recruiter Performance Dataset
recruiters = [fake.name() for _ in range(12)]
recruiter_data = []

for recruiter in recruiters:
    # Assign candidates to recruiters
    recruiter_candidates = random.sample(list(applications_df['candidate_id']), 
                                       random.randint(180, 300))
    
    total_applications = len(recruiter_candidates)
    hired_count = len([c for c in recruiter_candidates if c in hired_candidates])
    
    # Calculate metrics
    hire_rate = hired_count / total_applications if total_applications > 0 else 0
    avg_time_to_hire = np.mean([random.randint(25, 65) for _ in range(hired_count)]) if hired_count > 0 else 0
    
    recruiter_data.append({
        'recruiter_name': recruiter,
        'total_applications_sourced': total_applications,
        'total_hires': hired_count,
        'hire_rate': hire_rate,
        'avg_time_to_hire_days': avg_time_to_hire,
        'avg_candidate_satisfaction': np.random.uniform(3.2, 4.8),
        'linkedin_connections': random.randint(800, 3500),
        'years_experience': random.randint(1, 12)
    })

recruiters_df = pd.DataFrame(recruiter_data)

print("📋 Generating Candidate Survey Dataset...")

# Generate Candidate Survey Dataset (subset of candidates)
survey_candidates = random.sample(list(applications_df['candidate_id']), 487)
survey_data = []

for candidate_id in survey_candidates:
    app_info = applications_df[applications_df['candidate_id'] == candidate_id].iloc[0]
    was_hired = candidate_id in hired_candidates
    
    # Survey responses (1-5 scale)
    # Hired candidates generally rate experience higher
    if was_hired:
        overall_rating = np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6])
        communication_rating = np.random.choice([3, 4, 5], p=[0.15, 0.35, 0.5])
        process_speed_rating = np.random.choice([2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.3])
    else:
        overall_rating = np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
        communication_rating = np.random.choice([1, 2, 3, 4], p=[0.15, 0.25, 0.35, 0.25])
        process_speed_rating = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
    
    transparency_rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.3, 0.3, 0.15])
    
    # Would recommend company
    recommend_prob = 0.8 if overall_rating >= 4 else (0.4 if overall_rating == 3 else 0.1)
    would_recommend = random.random() < recommend_prob
    
    survey_data.append({
        'candidate_id': candidate_id,
        'survey_date': random_date(datetime(2024, 1, 1), datetime(2024, 6, 30)),
        'overall_experience_rating': overall_rating,
        'communication_rating': communication_rating,
        'process_speed_rating': process_speed_rating,
        'transparency_rating': transparency_rating,
        'would_recommend_company': would_recommend,
        'feedback_comments': fake.text(max_nb_chars=200) if random.random() > 0.7 else None
    })

survey_df = pd.DataFrame(survey_data)

# Save all datasets to CSV files
print("💾 Saving datasets to CSV files...")

applications_df.to_csv('ta_applications.csv', index=False)
pipeline_df.to_csv('ta_interview_pipeline.csv', index=False)
offers_df.to_csv('ta_offers.csv', index=False)
recruiters_df.to_csv('ta_recruiter_performance.csv', index=False)
survey_df.to_csv('ta_candidate_survey.csv', index=False)

print("\n" + "="*60)
print("✅ DATASET GENERATION COMPLETE!")
print("="*60)
print(f"📊 Applications Dataset: {len(applications_df):,} records")
print(f"🎯 Interview Pipeline: {len(pipeline_df):,} records")
print(f"💰 Offers Dataset: {len(offers_df):,} records")
print(f"👥 Recruiter Performance: {len(recruiters_df):,} records")
print(f"📋 Candidate Survey: {len(survey_df):,} records")

print("\n📁 Files Created:")
print("   • ta_applications.csv")
print("   • ta_interview_pipeline.csv")
print("   • ta_offers.csv")
print("   • ta_recruiter_performance.csv")
print("   • ta_candidate_survey.csv")

print("\n🎉 Ready for Analysis!")
print("   Run your analysis script to start the full TA analytics project.")
print("   All datasets are interconnected and ready for your portfolio.")

# Quick data quality summary
print("\n" + "="*60)
print("📈 DATASET SUMMARY")
print("="*60)

total_hires = len(pipeline_df[pipeline_df['outcome'] == 'Hired'])
overall_hire_rate = (total_hires / len(applications_df)) * 100

print(f"Overall Hire Rate: {overall_hire_rate:.1f}%")
print(f"Total Successful Hires: {total_hires:,}")

print("\nTop Sources by Volume:")
source_summary = applications_df['source'].value_counts().head(5)
for source, count in source_summary.items():
    print(f"   {source}: {count:,} applications")

print("\nDepartment Distribution:")
dept_summary = applications_df['department'].value_counts().head(5)
for dept, count in dept_summary.items():
    print(f"   {dept}: {count:,} applications")

print(f"\nDiversity Metrics (Applications):")
gender_dist = applications_df['gender'].value_counts(normalize=True) * 100
print(f"   Female: {gender_dist.get('Female', 0):.1f}%")
print(f"   Male: {gender_dist.get('Male', 0):.1f}%")

ethnic_underrep = applications_df[~applications_df['ethnicity'].isin(['White', 'Prefer not to say'])]
underrep_pct = (len(ethnic_underrep) / len(applications_df)) * 100
print(f"   Underrepresented Groups: {underrep_pct:.1f}%")

print("\n🚀 Your TA Analytics Project is Ready to Go!")
print("   Next step: Run the analysis script to generate insights and visualizations!")