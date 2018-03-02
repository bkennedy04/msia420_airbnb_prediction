import os
import pandas as pd
import math

class FeatureEngineer():
    def __init__(self):
        # self.session, self.train, self.test, self.age_gender, self.country, self.sample_submission = self.data_reader()
        pass

    def data_reader(self):
        """
        read data into notebook
        """

        data_dir = os.path.join('../..', 'data')

        session_path = os.path.join(data_dir, 'sessions.csv')
        train_path = os.path.join(data_dir, 'train_users_2.csv')
        test_path = os.path.join(data_dir, 'test_users.csv')
        age_gender_bkt_path = os.path.join(data_dir, 'age_gender_bkts.csv')
        country_path = os.path.join(data_dir, 'countries.csv')
        sample_submission_path = os.path.join(data_dir, 'sample_submission_NDF.csv')

        session = pd.read_csv(session_path)
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        age_gender = pd.read_csv(age_gender_bkt_path)
        country = pd.read_csv(country_path)
        sample_submission = pd.read_csv(sample_submission_path)

        return session, train, test, age_gender, country, sample_submission

    def age_transform(self, age):
        """
        Replace strange ages
        """
        current_year = 2014
        if (age > 2000):
            new_age = 34  # replace it with median
        elif (age > 1000):
            new_age = 2014 - age + 1
        elif (age > 100):
            new_age = 90
        elif (age < 10):
            new_age = 34  # replace it with median
        elif (math.isnan(age)):
            new_age = 34  # replace it with median
        else:
            new_age = age

        return new_age

    def train_cleaner(self, train):
        """
        data transformation for train dataframe
        """
        df = train.copy()

        # replace strange ages
        df['age'] = df['age'].apply(lambda x: self.age_transform(x))

        # replace value with most common values, which is "untracked"
        df['first_affiliate_tracked'] = df['first_affiliate_tracked'].fillna('untracked')

        # change column name id into user_id in order to merge with session data
        df = df.rename(columns={'id': 'user_id'})

        return df

    def session_cleaner(self, session):
        """
        data transformation for session dataframe
        """

        # drop rows where userid is null
        session_v1 = session.dropna(axis=0, subset=["user_id"])
        return session_v1

    def session_feature_engineer(self, session_v1):
        """
        feature engineer for cleaned session table
        """

        session_v2 = session_v1.groupby(['user_id']).agg({'user_id': 'count',
                                                          'action': 'nunique',
                                                          'secs_elapsed': 'sum',
                                                          'device_type': 'nunique',
                                                          }).rename(columns={'user_id': 'obs_count',
                                                                             'action': 'unique_action',
                                                                             'secs_elapsed': 'total_secs_elapsed',
                                                                             'device_type': 'unique_device'}).reset_index()
        session_v2['avg_time'] = session_v2["total_secs_elapsed"] / session_v2["obs_count"]

        session_v2 = session_v2[session_v2.total_secs_elapsed.notnull()]

        return session_v2

    def session_event_feature(self):
        pass


    def table_merger(self, train, session, drop_feature):
        """

        Args:
            train:
            session:
            drop_feature:

        Returns:

        """

        # Merge train with session
        train_session = pd.merge(session, train, on='user_id', how='inner')
        train_session['isNDF'] = [True if x == 'NDF' else False for x in train_session['country_destination']]

        train_session.drop(drop_feature, axis=1, inplace=True)

        return train_session

    def dummy_generator(self, train_session, category_list):
        """

        Args:
            train_session:
            category_list:

        Returns:

        """

        # Convert data type as 'category'
        for i in category_list:
            train_session[i] = train_session[i].astype('category')

        # Create dummy variables
        train_binary_dummy = pd.get_dummies(train_session, columns=category_list)

        return train_binary_dummy




if __name__=='__main__':
    fe_object = FeatureEngineer()
    session, train, test, age_gender, country, sample_submission = fe_object.data_reader()

    # Data Transformation
    train = fe_object.train_cleaner(train)
    session_v1 = fe_object.session_cleaner(session)
    session_v2 = fe_object.session_feature_engineer(session_v1)

    # Merge train with session
    drop_feature = ['user_id', 'total_secs_elapsed', 'date_account_created', 'timestamp_first_active',
                    'date_first_booking', 'country_destination']
    train_session = fe_object.table_merger(train, session_v2, drop_feature)


    # transform categorical feature into dummies for model building
    categorical = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
                   'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type',
                   'first_browser']

    train_binary_dummy = fe_object.dummy_generator(train_session, categorical)

    # # Save dataframe into csv
    # train_session.to_csv('../../data/train_binary.csv', index=False)
    # train_binary_dummy.to_csv('../../data/train_binary_dummy.csv', index=False)
