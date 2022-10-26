from training.generate_reason_code_distribution_on_populations import get_top_n_reason_codes
from training.inferred_amount_v2_ranked_amount import get_ranked_amount
from training.inferred_amount_v2_blended_amount import get_blended_amount
from training.inferred_amount_v2_experiment_3 import get_inferred_amount
from utils import buckets, naics_code, region, performance_lookups, zipcode_data_lookups
from datetime import datetime
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import re
import sys
from xgboost import XGBRegressor

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, DIR)


def get_exclusion_score_and_reason(paid_user, relation_flag):
    exclusion_score = 0
    exclusion_reason = "none"
    if paid_user == 1:
        exclusion_score = 1
        exclusion_reason = "paid account"
    elif relation_flag == 1:
        exclusion_score = 2
        exclusion_reason = "free account"
    return pd.Series([exclusion_score, exclusion_reason])


def get_tib_months(date):
    if pd.isnull(date):
        return np.nan
    date = datetime.strptime(date, '%Y-%m-%d')
    todays_date = datetime.today()
    month_difference = (todays_date.year - date.year) * \
        12 + todays_date.month - date.month + 1
    return month_difference


def get_naics_code_sector(naics_code_value):
    naics_code_sector = str(naics_code_value)[0:2]
    if (len(str(naics_code_value)) == 6 or len(str(naics_code_value)) == 4) and naics_code_sector in naics_code.valid_naics_code_sectors:
        return naics_code_sector
    return np.nan


def map_state_to_region(state):
    if pd.isnull(state):
        return np.nan
    elif state not in region.state_to_region_dict:
        return np.nan
    return region.state_to_region_dict[state]


def validate_zipcode(zipcode):
    zipcode = str(zipcode)
    if len(zipcode) == 5 and zipcode.isnumeric():
        return zipcode
    if len(zipcode) == 4 and zipcode.isnumeric():
        return '0' + zipcode
    if len(zipcode) == 3 and zipcode.isnumeric():
        return '00' + zipcode
    if len(zipcode) == 10 and re.match("^[0-9]{5}(?:-[0-9]{4})?$", zipcode):
        return zipcode[0:5]
    return np.nan


def get_num_employees_with_inference(num_employees):
    if pd.isnull(num_employees):
        return 1
    return num_employees


def get_naics_code_sector_performance_lookups(naics_code_sector):
    if naics_code_sector not in performance_lookups.naics_code_sector_verified_amount_gt0_25_percentile_rt_dict:
        naics_code_sector = np.nan
    naics_code_sector_verified_amount_gt0_25_percentile_rt = performance_lookups.naics_code_sector_verified_amount_gt0_25_percentile_rt_dict[
        naics_code_sector]
    naics_code_sector_verified_amount_gt0_median_rt = performance_lookups.naics_code_sector_verified_amount_gt0_median_rt_dict[
        naics_code_sector]
    return pd.Series([naics_code_sector_verified_amount_gt0_25_percentile_rt, naics_code_sector_verified_amount_gt0_median_rt])


def get_region_performance_lookups(region):
    if region not in performance_lookups.region_verified_amount_gt0_median_rt_dict:
        return np.nan
    region_verified_amount_gt0_median_rt = performance_lookups.region_verified_amount_gt0_median_rt_dict[
        region]
    return region_verified_amount_gt0_median_rt


def get_zipcode_reference_data_lookups(zip_code):
    if zip_code not in zipcode_data_lookups.income_household_median_dict:
        return pd.Series([np.nan, np.nan, np.nan])
    income_household_median = zipcode_data_lookups.income_household_median_dict[zip_code]
    home_value = zipcode_data_lookups.home_value_dict[zip_code]
    family_dual_income = zipcode_data_lookups.family_dual_income_dict[zip_code]
    return pd.Series([income_household_median, home_value, family_dual_income])


class ModelService:

    def __init__(self):
        """Initialize model service with features and models from Inferred Amount
        """

        self.features = [
            'time_in_business_months',
            'naics_code_sector_verified_amount_gt0_25_percentile_rt',
            'naics_code_sector_verified_amount_gt0_median_rt',
            'region_verified_amount_gt0_median_rt',
            'num_employees_with_inference',
            'income_household_median',
            'home_value',
            'family_dual_income',
        ]

        self.regression_type = "3-regression"

        self.blended_amount_coefficients = pd.read_csv(
            "inferred_amount_v2/blended_amount_coefficients.csv")

        self.scaler = StandardScaler()
        self.scaler = pickle.load(
            open("inferred_amount_v2/standardscaler.pkl", 'rb'))

        self.model = XGBRegressor()
        self.model = pickle.load(
            open("inferred_amount_v2/xgboost-model.pkl", 'rb'))

        self.linear_regression_models = []
        for i in range(len(buckets.amount_positive_bucket_values_3_level)):
            reg = LinearRegression()
            reg = pickle.load(
                open("inferred_amount_v2/linear-regression-model-bucket-" + str(i) + ".pkl", 'rb'))
            self.linear_regression_models.append(reg)

    def predict(self, dataframe):
        """Infers funded amount value of an SMB

        Args:
            X: The input CSV passed to the model service containing the following columns.
                - paid_user (int): The registered user flag, where 0 = registered user and
                    1 = not registered user
                - relation_flag (int): The relation flag, where 0 = direct user and 1 = embedded user
                - business_start_date (string): The business start date in 'YYYY-mm-dd' format.
                - naics_code (float): The 6-digit NAICS code of the business.
                - state (string): The 2-character abbreviated geographical state of the business.
                - zip_code (string): The 5-digit zip code of the business.
                - num_employees (int): The number of employees in the business.                
                - verified_amount (float): The lender verified annual gross revenue of the business in USD.
                - self_reported_amount (float): The self-reported annual gross revenue of the business in USD.
                - partner_amount (float): The partner reported annual gross revenue of the business in USD.
                - financial_aggregator_amount (float): The financial aggregator reported annual gross revenue of the business in USD.
                - bureau_amount_1 (float): Bureau1 reported annual gross revenue of the business in USD.
                - bureau_amount_2 (float): Bureau2  reported annual gross revenue of the business in USD.

        Returns:
            result: The output of the model service containing the following columns.
                - exclusion score (int): The score of the exclusion criteria
                - exclusion_reason (string): The description of the exclusion criteria.
                - inferred_amount_raw (float): The raw score from the model prior to calibration.
                - reason_1 (string): The top 1st reason why the SMB did not score higher.
                - reason_2 (string): The top 2nd reason why the SMB did not score higher.
                - reason_3 (string): The top 3rd reason why the SMB did not score higher.
                - reason_4 (string): The top 4th reason why the SMB did not score higher.
                - inferred_amount_calibrated (float): The calibrated score that is generated by transforming the
                    raw score to a more consumable value.                
                - ranked_amount (float): The amount that is ranked (by strength within a three bucket cohort).
                - ranked_amount_source (string): The amount source that is selected by the arbitration logic.
                - blended_amount (float): The amount that is blended through a series of regression calculations.                
        """

        input_fields = [
            'paid_user',
            'relation_flag',
            'business_start_date',
            'naics_code',
            'state',
            'zip_code',
            'num_employees',
            'verified_amount',
            'self_reported_amount',
            'partner_amount',
            'financial_aggregator_amount',
            'bureau_amount_1',
            'bureau_amount_2'
        ]
        dataframe = pd.DataFrame(dataframe.values, columns=input_fields)

        # get exclusion score and reason
        dataframe[['exclusion_score', 'exclusion_reason']] = dataframe.apply(lambda x: get_exclusion_score_and_reason(
            paid_user=x['paid_user'], relation_flag=x['relation_flag']), axis=1)

        # get inferred amount for users that are not excluded
        X = dataframe[dataframe['exclusion_score'] == 0]

        # feature calculation and validation
        X['time_in_business_months'] = X['business_start_date'].apply(
            lambda x: get_tib_months(date=x))
        X['naics_code_sector'] = X['naics_code'].apply(
            lambda x: get_naics_code_sector(naics_code_value=x))
        X['region'] = X['state'].apply(lambda x: map_state_to_region(state=x))
        X['zip_code'] = X['zip_code'].apply(lambda x: validate_zipcode(x))
        X['num_employees_with_inference'] = X['num_employees'].apply(
            lambda x: get_num_employees_with_inference(num_employees=x))

        # generate performance look-up features
        X[['naics_code_sector_verified_amount_gt0_25_percentile_rt', 'naics_code_sector_verified_amount_gt0_median_rt']] = \
            X['naics_code_sector'].apply(
                lambda x: get_naics_code_sector_performance_lookups(naics_code_sector=x))
        X['region_verified_amount_gt0_median_rt'] = X['region'].apply(
            lambda x: get_region_performance_lookups(region=x))

        # generate zip code reference data features
        X[['income_household_median', 'home_value', 'family_dual_income']] = X['zip_code'].apply(lambda x:
                                                                                                 get_zipcode_reference_data_lookups(zip_code=x))

        # get inferred amount
        X[['inferred_amount_raw', 'inferred_amount_calibrated']] = get_inferred_amount(dataframe=X,
                                                                                       features=self.features,
                                                                                       scaler=self.scaler,
                                                                                       model=self.model,
                                                                                       regression_type=self.regression_type,
                                                                                       linear_regression_models=self.linear_regression_models,
                                                                                       get_raw_inferred_amount=True)

        # get reason codes [4]
        X[['reason_1', 'reason_2', 'reason_3', 'reason_4']
          ] = get_top_n_reason_codes(X, 4)

        # get ranked amount
        X[['ranked_amount', 'ranked_amount_source']] = X.apply(lambda x: get_ranked_amount(
            x['verified_amount'],
            x['self_reported_amount'],
            x['partner_amount'],
            x['financial_aggregator_amount'],
            x['bureau_amount_1'],
            x['bureau_amount_2'],
            x['inferred_amount_calibrated'],
            get_source=True), axis=1)

        # get blended amount
        X['blended_amount'] = X.apply(lambda x: get_blended_amount(
            x['verified_amount'],
            x['self_reported_amount'],
            x['partner_amount'],
            x['financial_aggregator_amount'],
            x['bureau_amount_1'],
            x['bureau_amount_2'],
            x['inferred_amount_calibrated'],
            blended_amount_coefficients=self.blended_amount_coefficients), axis=1)

        dataframe = pd.merge(dataframe,
                             X,
                             how='left',
                             left_on=input_fields +
                                 ['exclusion_score', 'exclusion_reason'],
                             right_on=input_fields +
                                 ['exclusion_score', 'exclusion_reason']
                             )

        result = dataframe[[
            'time_in_business_months',
            'naics_code_sector',
            'region',
            'zip_code',
            'num_employees_with_inference',
            'naics_code_sector_verified_amount_gt0_25_percentile_rt',
            'naics_code_sector_verified_amount_gt0_median_rt',
            'region_verified_amount_gt0_median_rt',
            'income_household_median',
            'home_value',
            'family_dual_income',
            'exclusion_score',
            'exclusion_reason',
            'inferred_amount_raw',
            'reason_1',
            'reason_2',
            'reason_3',
            'reason_4',
            'inferred_amount_calibrated',
            'ranked_amount',
            'ranked_amount_source',
            'blended_amount',
        ]]
        return np.array(result)

    def health(self):
        if self.model:
            return True
        return False
